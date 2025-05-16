#include <stdlib.h>
#include "darknet.h"
#include "network.h"
#include "parser.h"
#include "detector.h"
#include "option_list.h"
#include "gpu_common.h"

#include <pthread.h>
#define _GNU_SOURCE
#include <sched.h>
#include <unistd.h>

#ifdef NVTX
#include "nvToolsExt.h"
#endif

#ifdef MEASURE
#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif
#endif

// GPU 전용 스레드 함수
static void* gpu_dedicated_thread(void* arg) {
    int core_id = sched_getcpu();
    if (VISUAL) printf("GPU-dedicated thread bound to core %d\n", core_id);
    
    // GPU 초기화 - 한 번만 실행
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
        CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    }
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
        CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
        
        // CUDA 초기화 상태 확인
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            printf("CUDA initialization error: %s\n", cudaGetErrorString(status));
        } else {
            if (VISUAL) printf("CUDA initialized successfully on device %d\n", gpu_index);
        }
    }

    // Pinned 메모리 할당 (초기화 시 한 번만)
    float* pinned_buffer = NULL;
    cudaError_t status = cudaMallocHost((void**)&pinned_buffer, MAX_BUFFER_SIZE * sizeof(float));
    if (status != cudaSuccess) {
        printf("Failed to allocate pinned memory: %s\n", cudaGetErrorString(status));
        // 오류 처리
    }
    
    gpu_task_t current_task;
    
    while (1) {
        // 작업 큐에서 작업 가져오기
        pthread_mutex_lock(&gpu_queue_mutex);
        while (gpu_task_head == gpu_task_tail) {
            if (VISUAL) printf("GPU Thread: Check gpu_task_head == gpu_task_tail for worker %d (layers %d-%d)\n", 
               current_task.thread_id, current_task.Gstart, current_task.Gend);
            pthread_cond_wait(&gpu_queue_cond, &gpu_queue_mutex);
        }
        
        current_task = gpu_task_queue[gpu_task_head % MAX_GPU_QUEUE_SIZE];
        gpu_task_head++;
        pthread_mutex_unlock(&gpu_queue_mutex);
        
        if (VISUAL) printf("GPU Thread: Processing task for worker %d (layers %d-%d)\n", 
               current_task.thread_id, current_task.Gstart, current_task.Gend);
        
        // H2D 복사 시작 시간 기록
        current_task.push_start_time = current_time_in_ms();
        
        // 실제 GPU 작업 수행 준비
        if (current_task.net.gpu_index != cuda_get_device())
            cuda_set_device(current_task.net.gpu_index);
        
        // GPU 메모리 확인 및 준비
        if (current_task.net.input_state_gpu == NULL) {
            printf("ERROR: GPU memory not allocated. Trying to allocate...\n");
            
            // GPU 메모리 할당 시도
            size_t size_bytes = current_task.size * sizeof(float);
            cudaError_t status = cudaMalloc((void**)&current_task.net.input_state_gpu, size_bytes);
            if (status != cudaSuccess) {
                printf("Failed to allocate GPU memory: %s\n", cudaGetErrorString(status));
                // 오류 처리
                continue;
            }
        }
        
        network_state state;
        state.index = 0;
        state.net = current_task.net;
        state.input = current_task.net.input_state_gpu;
        state.truth = 0;
        state.train = 0;
        state.delta = 0;
        
        // 입력 데이터를 GPU로 복사 (Gstart 레이어 입력)
    
        // 디버깅 출력 추가
        if (VISUAL) printf("Debug - GPU Thread: input=%p, state.input=%p, size=%d\n", current_task.input, state.input, current_task.size);
     
        // 입력 데이터를 pinned 메모리로 복사
        if (current_task.size * sizeof(float) <= MAX_BUFFER_SIZE * sizeof(float)) {
            memcpy(pinned_buffer, current_task.input, current_task.size * sizeof(float));

            // **수정된 부분: GPU 메모리로 정확한 위치에 복사**
            // Gstart > 0일 때, 입력 레이어의 input_gpu가 아니라 output_gpu를 입력으로 사용해야 합니다.
            if (current_task.Gstart == 0) {
                cuda_push_array(state.input, pinned_buffer, current_task.size);
            } else {
                // 이전 레이어의 output_gpu를 입력으로 사용
                int prev_layer_idx = current_task.Gstart - 1;
                layer prev_layer = current_task.net.layers[prev_layer_idx];
                cuda_push_array(prev_layer.output_gpu, pinned_buffer, current_task.size);
                state.input = prev_layer.output_gpu;  // 이 부분이 매우 중요!
            }
        } else {
            printf("ERROR: Buffer too small for data\n");
        }
        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
        
        // Skip connection 데이터도 함께 GPU로 복사
        if (current_task.Gstart > 0) {
            for (int i = 0; i < current_task.skip_count; i++) {
                int layer_idx = current_task.skip_layers_idx[i];
                layer skip_layer = current_task.net.layers[layer_idx];
                
                // CPU 데이터를 GPU로 복사
                if (current_task.skip_layers_size[i] * sizeof(float) <= MAX_BUFFER_SIZE * sizeof(float)) {
                    memcpy(pinned_buffer, current_task.skip_layers_data[i], 
                        current_task.skip_layers_size[i] * sizeof(float));
                    
                    // Pinned 메모리에서 GPU로 복사
                    cuda_push_array(skip_layer.output_gpu, pinned_buffer, current_task.skip_layers_size[i]);
                    
                    if (VISUAL) printf("GPU Thread: Copied skip connection data for layer %d\n", layer_idx);
                } else {
                    printf("ERROR: Buffer too small for skip connection data at layer %d\n", layer_idx);
                }
            }
        }

        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

        // H2D 복사 종료 시간 기록
        current_task.push_end_time = current_time_in_ms();
                
        // GPU 작업 시작 시간 기록
        current_task.gpu_start_time = current_time_in_ms();
        
        // GPU 작업 시작
        state.workspace = current_task.net.workspace;
        
        // GPU 실행 부분 (Gstart부터 Gend까지의 레이어 실행)
        for(int j = current_task.Gstart; j < current_task.Gend; ++j){
            state.index = j;
            layer l = current_task.net.layers[j];

            if(l.delta_gpu && state.train){
                fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
            }
            l.forward_gpu(l, state);
            state.input = l.output_gpu;
        }
        
        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
        
        // GPU 작업 종료 시간 기록
        current_task.gpu_end_time = current_time_in_ms();
        
        // D2H 복사 시작 시간 기록
        current_task.pull_start_time = current_time_in_ms();
        
        // 최종 레이어 결과만 가져오기
        layer final_layer = current_task.net.layers[current_task.Gend-1];
        cuda_pull_array(final_layer.output_gpu, final_layer.output, final_layer.outputs * final_layer.batch);
        
        // skipped_layers 처리 (필요한 경우에만)
        for(int i = current_task.Gend; i < current_task.net.n; i++) {
            for(int j = 0; j < 10; j++) {
                if((skip_layers[i][j] >= current_task.Gstart) && 
                   (skip_layers[i][j] < current_task.Gend) && 
                   (skip_layers[i][j] != 0)) {
                    int layer_idx = skip_layers[i][j];
                    layer skip_layer = current_task.net.layers[layer_idx];
                    cuda_pull_array(skip_layer.output_gpu, skip_layer.output, skip_layer.outputs * skip_layer.batch);
                }
            }
        }
        
        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
        
        // D2H 복사 종료 시간 기록
        current_task.pull_end_time = current_time_in_ms();
        
        // GPU 작업 로그 저장
        save_gpu_log(current_task);
        
        // 작업 완료 표시 및 워커 스레드에 알림
        pthread_mutex_lock(&result_mutex[current_task.task_id % MAX_GPU_QUEUE_SIZE]);
        current_task.completed = 1;
        gpu_task_queue[current_task.task_id % MAX_GPU_QUEUE_SIZE] = current_task;
        pthread_cond_signal(&result_cond[current_task.task_id % MAX_GPU_QUEUE_SIZE]);
        pthread_mutex_unlock(&result_mutex[current_task.task_id % MAX_GPU_QUEUE_SIZE]);
    }
    // 종료 시 해제
    if (pinned_buffer) cudaFreeHost(pinned_buffer);
    return NULL;
}

// 워커 스레드 함수 수정
static void threadFunc(thread_data_t data)
{
    // __Worker-thread-initialization__
    pthread_mutex_lock(&mutex_init);
    // GPU SETUP - 초기화만 수행, 실제 GPU 작업은 GPU 스레드가 담당
    list *options = read_data_cfg(data.datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size);
    char buff[256];
    char *input = buff;
    image **alphabet = load_alphabet();
    float nms = .45;
    double time;
    int top = 5;
    int index, i, j, k = 0;
    int* indexes = (int*)xcalloc(top, sizeof(int));
    int nboxes;
    detection *dets;
    image im, resized, cropped;
    float *X, *predictions;
    char *target_model = "yolo";
    int object_detection = strstr(data.cfgfile, target_model);
    int device = 1;
    extern gpu_yolo;
    network net = parse_network_cfg_custom(data.cfgfile, 1, 1, device);
    layer l = net.layers[net.n - 1];
    if (data.weightfile) {
        load_weights(&net, data.weightfile);
    }
    if (net.letter_box) data.letter_box = 1;
    net.benchmark_layers = data.benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    int skipped_layers[1000] = {0, };
    for(i = 0; i < net.n; i++) {
        for(j = 0; j < 10; j++) {
            if((skip_layers[i][j] != 0)) {
                skipped_layers[skip_layers[i][j]] = 1;
            }
        }
    }
    srand(2222222);
    if (data.filename) strncpy(input, data.filename, 256);
    else printf("Error! File is not exist.");

    if (data.thread_id == 1){
        print_layer_info(net);
        printf("num_pseudo_layer: %d\n", num_pseudo_layer);
    }
    int core_id = sched_getcpu();
    pthread_mutex_unlock(&mutex_init);

    pthread_barrier_wait(&barrier);
    // 각 워커별 GPU 사용 범위 설정
    // int Gstart = pseudo_layer_indexes[0];    // GPU 작업 시작 레이어 인덱스
    // int Gend = pseudo_layer_indexes[1];    // GPU 작업 종료 레이어 인덱스

    int Gstart = pseudo_layer_indexes[data.Gstart];
    int Gend = pseudo_layer_indexes[data.Gend];

    if (data.thread_id == 1) {
        // 로그 카운터 초기화
        gpu_log_count = 0;
        worker_log_count = 0;

        // 로그 배열 초기화 (선택적)
        memset(gpu_logs, 0, sizeof(gpu_logs));
        memset(worker_logs, 0, sizeof(worker_logs));
        printf("[Test] GPU-Accel with %d worker threads (GPU layers: %d-%d)\n", num_thread, Gstart, Gend);
    }

    // __Chekc-worker-thread-initialization__
    if (Gstart == Gend) {
        if (VISUAL) printf("\nThread %d is set to CPU core %d (CPU-only mode, no GPU layers)\n\n", data.thread_id, sched_getcpu());
    } else {
        if (VISUAL) printf("\nThread %d is set to CPU core %d (GPU layers: %d-%d)\n\n", data.thread_id, sched_getcpu(), Gstart, Gend);
    }
    pthread_barrier_wait(&barrier);

    for (i = 0; i < num_exp; i++) {
        if (i == START_IDX) pthread_barrier_wait(&barrier);

        // 워커 작업 시작 시간 기록
        double worker_start_time = current_time_in_ms();
        // __Preprocess__ (Pre-GPU 1)
        im = load_image(input, 0, 0, net.c);
        resized = resize_min(im, net.w);
        cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        X = cropped.data;
        double worker_inference_time = current_time_in_ms();
        
        // GPU를 사용하는 경우와 사용하지 않는 경우를 구분
        if (Gstart == Gend) {
            // GPU 사용 없이 CPU에서만 처리하는 경우
            double worker_request_time = current_time_in_ms();
            
            // 전체 네트워크를 CPU에서 실행
            network_state state;
            state.index = 0;
            state.net = net;
            state.input = X;
            state.truth = 0;
            state.train = 0;
            state.delta = 0;
            state.workspace = net.workspace_cpu;
            
            for(j = 0; j < net.n; ++j){
                state.index = j;
                l = net.layers[j];
            
                if(l.delta && state.train && l.train){
                    scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
                }
                l.forward(l, state);
    
                state.input = l.output;
            }
            
            
            double worker_receive_time = worker_request_time;
            double worker_postprocess_time = current_time_in_ms();
            
            predictions = get_network_output(net, 0);
            
            // 워커 로그 직접 저장 (CPU 전용 모드)
            worker_log_t worker_log;
            worker_log.thread_id = data.thread_id;
            worker_log.core_id = core_id;
            worker_log.Gstart = Gstart;
            worker_log.Gend = Gend;
            worker_log.worker_start_time = worker_start_time;
            worker_log.worker_inference_time = worker_inference_time;
            worker_log.worker_request_time = worker_request_time;
            worker_log.worker_receive_time = worker_receive_time;
            worker_log.worker_postprocess_time = worker_postprocess_time;
            worker_log.worker_end_time = 0; // 나중에 설정
            worker_log.push_time = 0;       // CPU 전용 모드에서는 0
            worker_log.compute_time = 0;    // CPU 전용 모드에서는 0
            worker_log.pull_time = 0;       // CPU 전용 모드에서는 0
            
            // __Postprecess__
            if (object_detection) {
                dets = get_network_boxes(&net, im.w, im.h, data.thresh, data.hier_thresh, 0, 1, &nboxes, data.letter_box);
                if (nms) {
                    if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                    else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
                }
                draw_detections_v3(im, dets, nboxes, data.thresh, names, alphabet, l.classes, data.ext_output);
            }
            else {
                if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
                top_k(predictions, net.outputs, top, indexes);
                for(j = 0; j < top; ++j){
                    index = indexes[j];
                    if (VISUAL) {
                        if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
                        else printf("[%d] %d thread %s: %f\n", i, data.thread_id, names[index], predictions[index]);
                    }
                }
            }
            
            // 워커 작업 종료 시간 기록
            double worker_end_time = current_time_in_ms();
            worker_log.worker_end_time = worker_end_time;
            
            save_worker_log(worker_log);
            
        } else {
            // GPU를 사용하는 경우 (수정된 부분)
            
            // 네트워크 상태 초기화
            int size = get_network_input_size(net) * net.batch;
            network_state pre_state;
            pre_state.index = 0;
            pre_state.net = net;
            pre_state.workspace = net.workspace_cpu;
            
            // Gstart가 0인 경우와 0이 아닌 경우 구분
            float *gpu_input = NULL;
            
        
            if (Gstart == 0) {
                // Gstart가 0인 경우: 원본 입력 데이터 사용
                
                // 입력 데이터 복사본 생성 (메모리 정렬 보장)
                size = get_network_input_size(net) * net.batch;
                gpu_input = (float*)malloc(size * sizeof(float));
                if (gpu_input) {
                    memcpy(gpu_input, X, size * sizeof(float));
                } else {
                    printf("ERROR: Failed to allocate memory for aligned copy\n");
                }
            } else {
                // Gstart가 0이 아닌 경우 - CPU에서 Gstart까지 처리
                pre_state.input = X;
                
                for(j = 0; j < Gstart; ++j){
                    pre_state.index = j;
                    l = net.layers[j];
                
                    if(l.delta && pre_state.train && l.train){
                        scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
                    }
                
                    l.forward(l, pre_state);
                
                    pre_state.input = l.output;
                }
                
                // 처리된 데이터 복사본 생성
                size = net.layers[Gstart - 1].outputs * net.batch;
                gpu_input = (float*)malloc(size * sizeof(float));
                if (gpu_input) {
                    memcpy(gpu_input, pre_state.input, size * sizeof(float));
                } else {
                    printf("ERROR: Failed to allocate memory for aligned copy\n");
                }
                
            }

            // GPU 작업 요청 준비
            int task_id;
            if (Gstart > 0) {
                // CPU에서 Gstart까지 처리한 후 skip connection 검사
                int skip_count = 0;

                
                // Gstart부터 Gend까지의 레이어들이 참조하는 모든 skip connection 확인
                for(int i = Gstart; i < Gend; i++) {
                    // 각 레이어의 skip connection 배열 검사
                    for(int j = 0; j < 10; j++) {
                        int skip_layer_idx = skip_layers[i][j];
                        
                        // 유효한 skip connection이고 Gstart 이전 레이어인 경우만 처리
                        if(skip_layer_idx > 0 && skip_layer_idx < Gstart) {
                            // 이미 추가된 skip connection인지 확인 (중복 방지)
                            bool already_added = false;
                            for(int k = 0; k < skip_count; k++) {
                                if(gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].skip_layers_idx[k] == skip_layer_idx) {
                                    already_added = true;
                                    break;
                                }
                            }
                            
                            // 아직 추가되지 않은 skip connection이면 추가
                            if(!already_added && skip_count < 10) {
                                gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].skip_layers_idx[skip_count] = skip_layer_idx;
                                gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].skip_layers_data[skip_count] = 
                                    net.layers[skip_layer_idx].output;
                                gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].skip_layers_size[skip_count] = 
                                    net.layers[skip_layer_idx].outputs * net.layers[skip_layer_idx].batch;
                                
                                skip_count++;
                            }
                        }
                    }
                }

                gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].skip_count = skip_count;
                if (VISUAL && skip_count > 0) printf("Worker %d: Added %d skip connections to GPU task\n", data.thread_id, skip_count);
            }
            // GPU 작업 요청 시간 기록
            double worker_request_time = current_time_in_ms();
            
            // GPU 작업 큐에 작업 추가
            pthread_mutex_lock(&gpu_queue_mutex);
            task_id = gpu_task_tail;
            
            // GPU 작업 정보 설정
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].input = gpu_input;

            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].size = size;
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].net = net;
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].task_id = task_id;
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].thread_id = data.thread_id;
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].core_id = core_id;
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].completed = 0;
            
            // GPU 작업 범위 설정
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].Gstart = Gstart;
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].Gend = Gend;
            
            // 시간 정보 설정
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].worker_start_time = worker_start_time;
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].worker_request_time = worker_request_time;
            gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].request_time = worker_request_time;
            
            // 입력 데이터는 GPU 스레드에서 직접 복사하도록 설정 (memcpy 제거)
            gpu_task_tail++;
            pthread_cond_signal(&gpu_queue_cond);
            pthread_mutex_unlock(&gpu_queue_mutex);
            
            if (VISUAL) printf("Worker %d: Requested GPU task %d (layers %d-%d)\n", data.thread_id, task_id, Gstart, Gend);
            
            
            // GPU 작업이 완료될 때까지 대기
            pthread_mutex_lock(&result_mutex[task_id % MAX_GPU_QUEUE_SIZE]);
            while (!gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE].completed) {
                pthread_cond_wait(&result_cond[task_id % MAX_GPU_QUEUE_SIZE], &result_mutex[task_id % MAX_GPU_QUEUE_SIZE]);
            }
            
            // GPU 결과 수신 시간 기록
            double worker_receive_time = current_time_in_ms();
            
            // GPU 작업 시간 정보 가져오기
            gpu_task_t completed_task = gpu_task_queue[task_id % MAX_GPU_QUEUE_SIZE];
            double push_time = completed_task.push_end_time - completed_task.push_start_time;
            double compute_time = completed_task.gpu_end_time - completed_task.gpu_start_time;
            double pull_time = completed_task.pull_end_time - completed_task.pull_start_time;
            
            // 메모리 해제 추가
            if (gpu_input) {
                free(gpu_input);
            }
            
            pthread_mutex_unlock(&result_mutex[task_id % MAX_GPU_QUEUE_SIZE]);
            
            if (VISUAL) printf("Worker %d: Received GPU result for task %d\n", data.thread_id, task_id);
            
            // CPU Inference (Post-GPU) - Gend부터 끝까지 CPU에서 처리
            network_state post_state;
            post_state.index = 0;
            post_state.net = net;
            post_state.input = net.layers[Gend-1].output;  // GPU에서 계산한 출력을 입력으로 사용
            post_state.workspace = net.workspace_cpu;
            gpu_yolo = 0;
            
            for(j = Gend; j < net.n; ++j){
                post_state.index = j;
                l = net.layers[j];

                if(l.delta && post_state.train && l.train){
                    scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
                }
                l.forward(l, post_state);

                post_state.input = l.output;
            }
            
            double worker_postprocess_time = current_time_in_ms();
            if (Gend == net.n) predictions = get_network_output_gpu(net);
            else predictions = get_network_output(net, 0);
            reset_wait_stream_events();

            // __Postprecess__ (Post-GPU 2)
            if (object_detection) {
                dets = get_network_boxes(&net, im.w, im.h, data.thresh, data.hier_thresh, 0, 1, &nboxes, data.letter_box);
                if (nms) {
                    if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                    else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
                }
                draw_detections_v3(im, dets, nboxes, data.thresh, names, alphabet, l.classes, data.ext_output);
            }
            else {
                if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
                top_k(predictions, net.outputs, top, indexes);
                for(j = 0; j < top; ++j){
                    index = indexes[j];
                    if (VISUAL) {
                        if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
                        else printf("[%d] %d thread %s: %f\n", i, data.thread_id, names[index], predictions[index]);
                    }
                }
            }

            // 워커 로그 직접 저장 (로컬 변수 사용)
            worker_log_t worker_log;
            worker_log.thread_id = data.thread_id;
            worker_log.core_id = core_id;
            worker_log.Gstart = Gstart;
            worker_log.Gend = Gend;
            worker_log.worker_start_time = worker_start_time;
            worker_log.worker_inference_time = worker_inference_time;
            worker_log.worker_request_time = worker_request_time;
            worker_log.worker_receive_time = worker_receive_time;
            worker_log.worker_postprocess_time = worker_postprocess_time;
            worker_log.push_time = push_time;
            worker_log.compute_time = compute_time;
            worker_log.pull_time = pull_time;
            
            // 워커 작업 종료 시간 기록
            double worker_end_time = current_time_in_ms();
            worker_log.worker_end_time = worker_end_time;

            save_worker_log(worker_log);
        }

        // free memory
        free_image(im);
        free_image(resized);
        free_image(cropped);
    }
    // 스레드 작업 완료 후 barrier에서 대기
    pthread_barrier_wait(&log_barrier);
    // thread_id가 1인 스레드만 로그 작성
    pthread_mutex_lock(&log_write_mutex);
    if (data.thread_id == 1) {
        char* model_name = malloc(strlen(data.cfgfile) + 1);
        strncpy(model_name, data.cfgfile + 6, (strlen(data.cfgfile)-10));
        model_name[strlen(data.cfgfile)-10] = '\0';

        char gpu_path[256];
        sprintf(gpu_path, "./measure/gpu-accel/%s/gpu_task_log/worker%d/G%d/gpu_task_log_G%d_%d.csv", model_name, num_thread, data.Gstart, data.Gstart, data.Gend);

        char worker_path[256];
        sprintf(worker_path, "./measure/gpu-accel/%s/worker_task_log/worker%d/G%d/worker_task_log_G%d_%d.csv", model_name, num_thread, data.Gstart, data.Gstart, data.Gend);

        // 로그 파일 작성
        write_logs_to_files(model_name, gpu_path, worker_path);
        if (VISUAL) printf("write_logs_to_files (GPU layers: %d-%d) --> worker_log_count: %d, gpu_log_count: %d\n", data.Gstart, data.Gend, worker_log_count, gpu_log_count);
        
        // 메모리 해제
        free(model_name);
    }
    pthread_mutex_unlock(&log_write_mutex);
    
    // free memory
    free_detections(dets, nboxes);
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_alphabet(alphabet);
    pthread_exit(NULL);
}

void gpu_accel(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int theoretical_exp, int theo_thread, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{

    if (MAXCORES < num_thread) {
    	printf("Error! Too many CPU cores!\n");
    	return 0;
    }

    int rc;
    int i;
    pthread_t gpu_thread;
    pthread_t threads[num_thread];
    thread_data_t data[num_thread];

    // 로그 카운터 초기화
    gpu_log_count = 0;
    worker_log_count = 0;

    // 결과 동기화 객체 초기화
    for (i = 0; i < MAX_GPU_QUEUE_SIZE; i++) {
        pthread_mutex_init(&result_mutex[i], NULL);
        pthread_cond_init(&result_cond[i], NULL);
    }

    // GPU 전용 스레드 생성
    rc = pthread_create(&gpu_thread, NULL, gpu_dedicated_thread, NULL);
    if (rc) {
        printf("Error: Unable to create GPU thread, %d\n", rc);
        exit(-1);
    }

    // GPU 스레드를 코어 3에 고정
    cpu_set_t gpu_cpuset;
    CPU_ZERO(&gpu_cpuset);
    CPU_SET(3, &gpu_cpuset);
    rc = pthread_setaffinity_np(gpu_thread, sizeof(gpu_cpuset), &gpu_cpuset);
    if (rc != 0) {
        fprintf(stderr, "GPU thread: pthread_setaffinity_np() failed\n");
        exit(0);
    }
    
    // 워커 스레드 배리어 초기화
    pthread_barrier_init(&barrier, NULL, num_thread);
    pthread_barrier_init(&log_barrier, NULL, num_thread);
    
    // 워커 스레드 생성
    for (i = 0; i < num_thread; i++) {
        data[i].datacfg = datacfg;
        data[i].cfgfile = cfgfile;
        data[i].num_thread = num_thread;
        data[i].weightfile = weightfile;
        data[i].filename = filename;
        data[i].thresh = thresh;
        data[i].hier_thresh = hier_thresh;
        data[i].dont_show = dont_show;
        data[i].ext_output = ext_output;
        data[i].save_labels = save_labels;
        data[i].outfile = outfile;
        data[i].letter_box = letter_box;
        data[i].benchmark_layers = benchmark_layers;
        data[i].thread_id = i + 1;
        data[i].Gstart = Gstart;
        data[i].Gend = Gend;
        rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
        if (rc) {
            printf("Error: Unable to create thread, %d\n", rc);
            exit(-1);
        }

        // __CPU AFFINITY SETTING__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(coreIDOrder[i], &cpuset); // 코어 할당 (3은 GPU 스레드용, 0, 1, 2은 OS 작업용)
        
        int ret = pthread_setaffinity_np(threads[i], sizeof(cpuset), &cpuset);
        if (ret != 0) {
            fprintf(stderr, "Worker thread: pthread_setaffinity_np() failed\n");
            exit(0);
        } 
    }

    // 워커 스레드 종료 대기
    for (i = 0; i < num_thread; i++) {
        pthread_join(threads[i], NULL);
    }

    // GPU 스레드 종료
    pthread_cancel(gpu_thread);
    pthread_join(gpu_thread, NULL);
    
    if (VISUAL) printf("Logs written to files\n");

    // 동기화 객체 정리
    for (i = 0; i < MAX_GPU_QUEUE_SIZE; i++) {
        pthread_mutex_destroy(&result_mutex[i]);
        pthread_cond_destroy(&result_cond[i]);
    }
    pthread_mutex_destroy(&gpu_queue_mutex);
    pthread_cond_destroy(&gpu_queue_cond);
    pthread_mutex_destroy(&log_mutex);
    pthread_barrier_destroy(&barrier);
    pthread_barrier_destroy(&log_barrier);

    return 0;
}
