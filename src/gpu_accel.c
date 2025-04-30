#include <stdlib.h>
#include "darknet.h"
#include "network.h"
#include "parser.h"
#include "detector.h"
#include "option_list.h"

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

#define START_IDX 3
#define VISUAL 0
#define MAX_BUFFER_SIZE 2097152

// 로그 쓰기를 위한 barrier와 뮤텍스
pthread_barrier_t log_barrier;
pthread_mutex_t log_write_mutex = PTHREAD_MUTEX_INITIALIZER;
int log_written = 0;  // 로그가 이미 작성되었는지 확인하는 플래그

// 시간 측정 함수
double current_time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// 로그 파일 핸들
FILE *fp_gpu;
FILE *fp_worker;

// 기존 동기화 객체
pthread_barrier_t barrier;
static pthread_mutex_t mutex_init = PTHREAD_MUTEX_INITIALIZER;

int skip_layers[1000][10] = {0};
static pthread_mutex_t mutex_gpu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static int current_thread = 1;

// GPU 작업 큐 관련 구조체 및 변수
#define MAX_GPU_QUEUE_SIZE 128
#define MAX_TASKS 1000  // 최대 작업 수 (로그 저장용)

typedef struct gpu_task_t {
    float *input;              // 입력 데이터
    int size;                  // 입력 데이터 크기
    network net;               // 네트워크 정보
    int task_id;               // 작업 ID
    int thread_id;             // 요청한 스레드 ID
    int completed;             // 완료 여부 플래그
    float *output;             // 출력 데이터가 저장될 위치
    
    // GPU 작업 범위 설정
    int Gstart;                // GPU 작업 시작 레이어 인덱스
    int Gend;                  // GPU 작업 종료 레이어 인덱스
    
    // 시간 측정을 위한 필드
    double request_time;       // 요청 시간
    double worker_start_time;  // 워커 작업 시작 시간
    double worker_request_time;// 워커가 GPU에 요청한 시간
    double worker_receive_time;// 워커가 GPU 결과를 받은 시간
    double worker_end_time;    // 워커 작업 종료 시간
    
    // 추가 시간 측정을 위한 필드
    double push_start_time;    // GPU 메모리로 복사 시작 시간
    double push_end_time;      // GPU 메모리로 복사 완료 시간
    double gpu_start_time;     // GPU 작업 시작 시간
    double gpu_end_time;       // GPU 작업 종료 시간
    double pull_start_time;    // GPU 메모리에서 복사 시작 시간
    double pull_end_time;      // GPU 메모리에서 복사 완료 시간

    // Skip connection을 위한 추가 필드
    int skip_count;                 // skip connection 개수
    int skip_layers_idx[10];        // skip connection 레이어 인덱스
    float *skip_layers_data[10];    // 각 skip connection 레이어의 데이터 포인터
    int skip_layers_size[10];       // 각 skip connection 레이어의 데이터 크기
} gpu_task_t;

// 로그 저장용 구조체
typedef struct gpu_log_t {
    int thread_id;
    int Gstart;                // GPU 시작 레이어
    int Gend;                  // GPU 종료 레이어
    double request_time;
    double push_start_time;
    double push_end_time;
    double gpu_start_time;
    double gpu_end_time;
    double pull_start_time;
    double pull_end_time;
} gpu_log_t;

typedef struct worker_log_t {
    int thread_id;
    int Gstart;                // GPU 시작 레이어
    int Gend;                  // GPU 종료 레이어
    double worker_start_time;
    double worker_request_time;
    double worker_receive_time;
    double worker_end_time;
    // 추가된 전송 시간 (워커 로그에도 GPU 작업 관련 시간 추가)
    double push_time;          // GPU 메모리로 전송 시간
    double compute_time;       // GPU 계산 시간
    double pull_time;          // GPU 메모리에서 전송 시간
} worker_log_t;

// 로그 저장용 배열
gpu_log_t gpu_logs[MAX_TASKS];
worker_log_t worker_logs[MAX_TASKS];
int gpu_log_count = 0;
int worker_log_count = 0;
pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

gpu_task_t gpu_task_queue[MAX_GPU_QUEUE_SIZE];
int gpu_task_head = 0, gpu_task_tail = 0;
pthread_mutex_t gpu_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t gpu_queue_cond = PTHREAD_COND_INITIALIZER;

// 결과 동기화를 위한 뮤텍스와 조건 변수
pthread_mutex_t result_mutex[MAX_GPU_QUEUE_SIZE];
pthread_cond_t result_cond[MAX_GPU_QUEUE_SIZE];

// 로그 함수 - 배열에 저장
void save_gpu_log(gpu_task_t task) {
    pthread_mutex_lock(&log_mutex);
    if (gpu_log_count < MAX_TASKS) {
        gpu_logs[gpu_log_count].thread_id = task.thread_id;
        gpu_logs[gpu_log_count].Gstart = task.Gstart;
        gpu_logs[gpu_log_count].Gend = task.Gend;
        gpu_logs[gpu_log_count].request_time = task.request_time;
        gpu_logs[gpu_log_count].push_start_time = task.push_start_time;
        gpu_logs[gpu_log_count].push_end_time = task.push_end_time;
        gpu_logs[gpu_log_count].gpu_start_time = task.gpu_start_time;
        gpu_logs[gpu_log_count].gpu_end_time = task.gpu_end_time;
        gpu_logs[gpu_log_count].pull_start_time = task.pull_start_time;
        gpu_logs[gpu_log_count].pull_end_time = task.pull_end_time;
        gpu_log_count++;
    }
    pthread_mutex_unlock(&log_mutex);
}

void save_worker_log(worker_log_t log) {
    pthread_mutex_lock(&log_mutex);
    if (worker_log_count < MAX_TASKS) {
        worker_logs[worker_log_count] = log;
        worker_log_count++;
    }
    pthread_mutex_unlock(&log_mutex);
}

// 정렬 비교 함수
int compare_gpu_logs(const void *a, const void *b) {
    gpu_log_t *log_a = (gpu_log_t *)a;
    gpu_log_t *log_b = (gpu_log_t *)b;
    if (log_a->request_time < log_b->request_time) return -1;
    if (log_a->request_time > log_b->request_time) return 1;
    return 0;
}

int compare_worker_logs(const void *a, const void *b) {
    worker_log_t *log_a = (worker_log_t *)a;
    worker_log_t *log_b = (worker_log_t *)b;
    if (log_a->worker_start_time < log_b->worker_start_time) return -1;
    if (log_a->worker_start_time > log_b->worker_start_time) return 1;
    return 0;
}

// 로그 파일 작성 함수
void write_logs_to_files(char *model_name, char *gpu_path, char *worker_path) {
    // GPU 로그 정렬 및 파일 작성
    qsort(gpu_logs, gpu_log_count, sizeof(gpu_log_t), compare_gpu_logs);

    fp_gpu = fopen(gpu_path, "w");
    if (!fp_gpu) {
        perror("파일 열기 실패");
        exit(1);
    }

    fprintf(fp_gpu, "thread_id,Gstart,Gend,request_time,push_start_time,push_end_time,gpu_start_time,gpu_end_time,pull_start_time,pull_end_time,push_delay,compute_delay,pull_delay,total_delay\n");
    for (int i = 0; i < gpu_log_count; i++) {
        double push_delay = gpu_logs[i].push_end_time - gpu_logs[i].push_start_time;
        double compute_delay = gpu_logs[i].gpu_end_time - gpu_logs[i].gpu_start_time;
        double pull_delay = gpu_logs[i].pull_end_time - gpu_logs[i].pull_start_time;
        double total_delay = gpu_logs[i].pull_end_time - gpu_logs[i].push_start_time;
        
        fprintf(fp_gpu, "%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                gpu_logs[i].thread_id,
                gpu_logs[i].Gstart,
                gpu_logs[i].Gend,
                gpu_logs[i].request_time, 
                gpu_logs[i].push_start_time, 
                gpu_logs[i].push_end_time,
                gpu_logs[i].gpu_start_time, 
                gpu_logs[i].gpu_end_time,
                gpu_logs[i].pull_start_time,
                gpu_logs[i].pull_end_time,
                push_delay,
                compute_delay,
                pull_delay,
                total_delay);
    }
    fclose(fp_gpu);
    
    // 워커 로그 정렬 및 파일 작성
    qsort(worker_logs, worker_log_count, sizeof(worker_log_t), compare_worker_logs);

    fp_worker = fopen(worker_path, "w");
    if (!fp_worker) {
        perror("파일 열기 실패");
        exit(1);
    }

    fprintf(fp_worker, "thread_id,Gstart,Gend,worker_start_time,worker_request_time,worker_receive_time,worker_end_time,push_time,compute_time,pull_time,total_gpu_time,preprocessing_time,postprocessing_time,total_time\n");
    for (int i = 0; i < worker_log_count; i++) {
        double preprocessing_time = worker_logs[i].worker_request_time - worker_logs[i].worker_start_time;
        double postprocessing_time = worker_logs[i].worker_end_time - worker_logs[i].worker_receive_time;
        double total_time = worker_logs[i].worker_end_time - worker_logs[i].worker_start_time;
        double total_gpu_time = worker_logs[i].push_time + worker_logs[i].compute_time + worker_logs[i].pull_time;
        
        fprintf(fp_worker, "%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                worker_logs[i].thread_id,
                worker_logs[i].Gstart,
                worker_logs[i].Gend,
                worker_logs[i].worker_start_time, 
                worker_logs[i].worker_request_time, 
                worker_logs[i].worker_receive_time, 
                worker_logs[i].worker_end_time,
                worker_logs[i].push_time,
                worker_logs[i].compute_time,
                worker_logs[i].pull_time,
                total_gpu_time,
                preprocessing_time,
                postprocessing_time,
                total_time);
    }
    fclose(fp_worker);
}

// 스레드 데이터 구조체
typedef struct thread_data_t{
    char *datacfg;
    char *cfgfile;
    char *weightfile;
    char *filename;
    float thresh;
    float hier_thresh;
    int dont_show;
    int ext_output;
    int save_labels;
    char *outfile;
    int letter_box;
    int benchmark_layers;
    int thread_id;
    int num_thread;
    bool isTest;
    // GPU 작업 범위 설정
    int Gstart;                // GPU 작업 시작 레이어 인덱스
    int Gend;                  // GPU 작업 종료 레이어 인덱스
} thread_data_t;

// GPU 전용 스레드 함수
void* gpu_dedicated_thread(void* arg) {
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
    extern int skip_layers[1000][10];
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
    pthread_mutex_unlock(&mutex_init);


    for (int s = 0; s < net.n; s++){
        for (int e = s + 1; s < net.n; e++){
            pthread_barrier_wait(&barrier);
            // 각 워커별 GPU 사용 범위 설정
            int Gstart = s;    // GPU 작업 시작 레이어 인덱스
            int Gend = e;    // GPU 작업 종료 레이어 인덱스

            if (data.thread_id == 1) {
                // 로그 카운터 초기화
                gpu_log_count = 0;
                worker_log_count = 0;

                // 로그 배열 초기화 (선택적)
                memset(gpu_logs, 0, sizeof(gpu_logs));
                memset(worker_logs, 0, sizeof(worker_logs));
                printf("GPU-Accel with %d worker threads (GPU layers: %d-%d)\n", num_thread, Gstart, Gend);
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
                    
                    predictions = get_network_output(net, 0);
                    
                    // 워커 로그 직접 저장 (CPU 전용 모드)
                    worker_log_t worker_log;
                    worker_log.thread_id = data.thread_id;
                    worker_log.Gstart = Gstart;
                    worker_log.Gend = Gend;
                    worker_log.worker_start_time = worker_start_time;
                    worker_log.worker_request_time = worker_request_time;
                    worker_log.worker_receive_time = worker_receive_time;
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
                    worker_log.Gstart = Gstart;
                    worker_log.Gend = Gend;
                    worker_log.worker_start_time = worker_start_time;
                    worker_log.worker_request_time = worker_request_time;
                    worker_log.worker_receive_time = worker_receive_time;
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
                sprintf(gpu_path, "./measure/gpu-accel/%s/gpu_task_log/worker%d/G%d/gpu_task_log_G%d_%d.csv", model_name, num_thread, Gstart, Gstart, Gend);

                char worker_path[256];
                sprintf(worker_path, "./measure/gpu-accel/%s/worker_task_log/worker%d/G%d/worker_task_log_G%d_%d.csv", model_name, num_thread, Gstart, Gstart, Gend);

                // 로그 파일 작성
                write_logs_to_files(model_name, gpu_path, worker_path);
                if (VISUAL) printf("write_logs_to_files (GPU layers: %d-%d) --> worker_log_count: %d, gpu_log_count: %d\n", Gstart, Gend, worker_log_count, gpu_log_count);
                
                // 메모리 해제
                free(model_name);
            }
            pthread_mutex_unlock(&log_write_mutex);
        }
    }
    
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

    // GPU 스레드를 코어 1에 고정
    cpu_set_t gpu_cpuset;
    CPU_ZERO(&gpu_cpuset);
    CPU_SET(1, &gpu_cpuset);
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
        data[i].Gstart = 0;
        data[i].Gend = 0;
        rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
        if (rc) {
            printf("Error: Unable to create thread, %d\n", rc);
            exit(-1);
        }

        // __CPU AFFINITY SETTING__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i + 2, &cpuset); // 코어 할당 (1부터 시작, 1은 GPU 스레드용, 0은 OS 작업용)
        
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