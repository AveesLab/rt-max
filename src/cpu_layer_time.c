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
#define VISUAL 1

static int coreIDOrder[MAXCORES] = {4, 5, 6, 7, 8, 9, 10, 11};

// 로그 쓰기를 위한 barrier와 뮤텍스
static pthread_barrier_t log_barrier;
static pthread_mutex_t log_write_mutex = PTHREAD_MUTEX_INITIALIZER;
static int log_written = 0;  // 로그가 이미 작성되었는지 확인하는 플래그

// 시간 측정 함수
static double current_time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// 로그 파일 핸들
static FILE *fp_worker;

// 기존 동기화 객체
static pthread_barrier_t barrier;
static pthread_mutex_t mutex_init = PTHREAD_MUTEX_INITIALIZER;

// 작업 로그 관련 구조체 및 변수
#define MAX_TASKS 1000  // 최대 작업 수 (로그 저장용)

// worker_log_t 구조체에 레이어별 시간 추가
typedef struct worker_log_t {
    int thread_id;
    int core_id;
    double worker_start_time;
    double worker_preprocess_end_time;
    double *layer_times;  // 각 레이어의 실행 시간을 저장할 배열
    int num_layers;       // 레이어 수
    double worker_inference_end_time;
    double worker_postprocess_end_time;
    double worker_end_time;
} worker_log_t;

// 로그 저장용 배열
static worker_log_t worker_logs[MAX_TASKS];
static int worker_log_count = 0;
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

// 로그 함수 - 배열에 저장 (수정)
static void save_worker_log(worker_log_t log) {
    pthread_mutex_lock(&log_mutex);
    if (worker_log_count < MAX_TASKS) {
        // 깊은 복사를 위한 메모리 할당
        worker_logs[worker_log_count].thread_id = log.thread_id;
        worker_logs[worker_log_count].core_id = log.core_id;
        worker_logs[worker_log_count].worker_start_time = log.worker_start_time;
        worker_logs[worker_log_count].worker_preprocess_end_time = log.worker_preprocess_end_time;
        worker_logs[worker_log_count].worker_inference_end_time = log.worker_inference_end_time;
        worker_logs[worker_log_count].worker_postprocess_end_time = log.worker_postprocess_end_time;
        worker_logs[worker_log_count].worker_end_time = log.worker_end_time;
        worker_logs[worker_log_count].num_layers = log.num_layers;
        
        // layer_times 배열 깊은 복사
        worker_logs[worker_log_count].layer_times = (double*)malloc(sizeof(double) * log.num_layers);
        if (worker_logs[worker_log_count].layer_times) {
            memcpy(worker_logs[worker_log_count].layer_times, log.layer_times, sizeof(double) * log.num_layers);
        }
        
        worker_log_count++;
    }
    pthread_mutex_unlock(&log_mutex);
}

// 정렬 비교 함수
static int compare_worker_logs(const void *a, const void *b) {
    worker_log_t *log_a = (worker_log_t *)a;
    worker_log_t *log_b = (worker_log_t *)b;
    if (log_a->worker_start_time < log_b->worker_start_time) return -1;
    if (log_a->worker_start_time > log_b->worker_start_time) return 1;
    return 0;
}

// 로그 파일 작성 함수 (수정)
static void write_logs_to_file(char *model_name, char *worker_path, int num_layers) {
    // 워커 로그 정렬 및 파일 작성
    qsort(worker_logs, worker_log_count, sizeof(worker_log_t), compare_worker_logs);

    fp_worker = fopen(worker_path, "w");
    if (!fp_worker) {
        perror("파일 열기 실패");
        exit(1);
    }

    // 워커 CSV 헤더 - 기본 정보 먼저 출력
    fprintf(fp_worker, "thread_id,core_id,worker_start_time,worker_preprocess_end_time,worker_inference_end_time,worker_postprocess_end_time,worker_end_time,preprocess_delay,inference_delay,postprocess_delay,total_delay");
    
    // 레이어별 열을 마지막에 추가
    for (int l = 0; l < num_layers; l++) {
        fprintf(fp_worker, ",layer%d_time", l);
    }
    fprintf(fp_worker, "\n");
    
    for (int i = 0; i < worker_log_count; i++) {
        // 워커 지연 시간 계산
        double preprocess_delay = worker_logs[i].worker_preprocess_end_time - worker_logs[i].worker_start_time;
        double inference_delay = worker_logs[i].worker_inference_end_time - worker_logs[i].worker_preprocess_end_time;
        double postprocess_delay = worker_logs[i].worker_postprocess_end_time - worker_logs[i].worker_inference_end_time;
        double total_delay = worker_logs[i].worker_end_time - worker_logs[i].worker_start_time;
        
        // 기본 정보 먼저 출력
        fprintf(fp_worker, "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f", 
            worker_logs[i].thread_id,
            worker_logs[i].core_id,
            worker_logs[i].worker_start_time, 
            worker_logs[i].worker_preprocess_end_time,
            worker_logs[i].worker_inference_end_time, 
            worker_logs[i].worker_postprocess_end_time, 
            worker_logs[i].worker_end_time,
            preprocess_delay,
            inference_delay,
            postprocess_delay,
            total_delay);
        
        // 레이어별 시간을 마지막에 출력
        for (int l = 0; l < worker_logs[i].num_layers; l++) {
            fprintf(fp_worker, ",%.2f", worker_logs[i].layer_times[l]);
        }
        fprintf(fp_worker, "\n");
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
} thread_data_t;

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
    srand(2222222);
    if (data.filename) strncpy(input, data.filename, 256);
    else printf("Error! File is not exist.");
    int core_id = sched_getcpu();
    pthread_mutex_unlock(&mutex_init);

    if (data.thread_id == 1) {
        // 로그 카운터 초기화
        worker_log_count = 0;

        // 로그 배열 초기화 (선택적)
        memset(worker_logs, 0, sizeof(worker_logs));
        printf("Measure CPU Layer Time with %d worker threads\n", num_thread);
    }

    // __Check-worker-thread-initialization__
    if (VISUAL) printf("\nThread %d is set to CPU core %d (CPU-only mode, no GPU layers)\n\n", data.thread_id, sched_getcpu());
    pthread_barrier_wait(&barrier);

    for (i = 0; i < num_exp; i++) {
        if (i == START_IDX) pthread_barrier_wait(&barrier);

        // 워커 작업 시작 시간 기록
        double worker_start_time = current_time_in_ms();
        
        // __Preprocess__
        im = load_image(input, 0, 0, net.c);
        resized = resize_min(im, net.w);
        cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        X = cropped.data;
        
        // 전처리 완료 시간 기록
        double worker_preprocess_end_time = current_time_in_ms();
        
        // 전체 네트워크를 CPU에서 실행
        network_state state;
        state.index = 0;
        state.net = net;
        state.input = X;
        state.truth = 0;
        state.train = 0;
        state.delta = 0;
        state.workspace = net.workspace_cpu;

        // 워커 로그 초기화 (레이어 시간 배열 할당)
        worker_log_t worker_log;
        worker_log.thread_id = data.thread_id;
        worker_log.core_id = core_id;
        worker_log.worker_start_time = worker_start_time;
        worker_log.worker_preprocess_end_time = worker_preprocess_end_time;
        worker_log.num_layers = net.n;
        worker_log.layer_times = (double*)malloc(sizeof(double) * net.n);
        
        if (!worker_log.layer_times) {
            fprintf(stderr, "Memory allocation failed for layer_times\n");
            exit(1);
        }
            
        // 레이어별 실행 시간 측정
        for(j = 0; j < net.n; ++j){
            // 레이어 시작 시간 측정
            double layer_start_time = current_time_in_ms();

            state.index = j;
            l = net.layers[j];
            
            if(l.delta && state.train && l.train){
                scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
            }
            
            // 레이어 실행
            l.forward(l, state);
            state.input = l.output;

            // 레이어 종료 시간 측정 및 저장
            double layer_end_time = current_time_in_ms();
            worker_log.layer_times[j] = layer_end_time - layer_start_time;
        }
        
        predictions = get_network_output(net, 0);

        // 추론 완료 시간 기록
        double worker_inference_end_time = current_time_in_ms();
        worker_log.worker_inference_end_time = worker_inference_end_time;
        
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
        
        // 후처리 완료 시간 기록
        double worker_postprocess_end_time = current_time_in_ms();
        
        // 워커 작업 종료 시간 기록
        double worker_end_time = current_time_in_ms();
        worker_log.worker_postprocess_end_time = worker_postprocess_end_time;
        worker_log.worker_end_time = worker_end_time;
        
        // 로그 저장 및 메모리 해제
        save_worker_log(worker_log);
        free(worker_log.layer_times);  // 원본 로그의 메모리 해제

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

        char worker_path[256];
        sprintf(worker_path, "./measure/layer_time/%s/worker%d/cpu_layer_time.csv", model_name, num_thread);

        // 디렉토리 생성 확인 (디렉토리가 없을 수 있음)
        char dir_cmd[512];
        sprintf(dir_cmd, "mkdir -p ./measure/layer_time/%s/worker%d", model_name, num_thread);
        system(dir_cmd);

        // 로그 파일 작성 (레이어 수 전달)
        write_logs_to_file(model_name, worker_path, net.n);
        if (VISUAL) printf("write_logs_to_file --> worker_log_count: %d\n", worker_log_count);
        
        // 메모리 해제
        free(model_name);
        
        // 로그에 사용된 모든 메모리 해제
        for (int j = 0; j < worker_log_count; j++) {
            if (worker_logs[j].layer_times) {
                free(worker_logs[j].layer_times);
                worker_logs[j].layer_times = NULL;
            }
        }
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

void cpu_layer_time(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    if (MAXCORES < num_thread) {
    	printf("Error! Too many CPU cores!\n");
    	return 0;
    }
    
    int rc;
    int i;
    pthread_t threads[num_thread];
    thread_data_t data[num_thread];

    // 로그 카운터 초기화
    worker_log_count = 0;
    
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
        
        rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
        if (rc) {
            printf("Error: Unable to create thread, %d\n", rc);
            exit(-1);
        }

        // __CPU AFFINITY SETTING__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(coreIDOrder[i], &cpuset); // 코어 할당 (0은 OS 작업용)
        
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
    
    if (VISUAL) printf("Logs written to files\n");

    // 동기화 객체 정리
    pthread_mutex_destroy(&log_mutex);
    pthread_barrier_destroy(&barrier);
    pthread_barrier_destroy(&log_barrier);

    return 0;
}