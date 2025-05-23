#include "gpu_common.h"
#include "detector.h"

int coreIDOrder[MAXCORES] = {4, 5, 6, 7, 8, 9, 10, 11};

int pseudo_layer_indexes[500];
int num_pseudo_layer = 0;

// 로그 쓰기를 위한 barrier와 뮤텍스
pthread_barrier_t log_barrier;
pthread_mutex_t log_write_mutex = PTHREAD_MUTEX_INITIALIZER;
int log_written = 0;  // 로그가 이미 작성되었는지 확인하는 플래그

// 로그 파일
FILE *fp_gpu;
FILE *fp_worker;

// 기존 동기화 객체
pthread_barrier_t barrier;
pthread_mutex_t mutex_init = PTHREAD_MUTEX_INITIALIZER;

pthread_mutex_t mutex_gpu = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int current_thread = 1;

// GPU 작업 큐 관련 변수
gpu_task_t gpu_task_queue[MAX_GPU_QUEUE_SIZE];
int gpu_task_head = 0, gpu_task_tail = 0;
pthread_mutex_t gpu_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t gpu_queue_cond = PTHREAD_COND_INITIALIZER;

// 로그 저장용 배열
gpu_log_t gpu_logs[MAX_TASKS];
worker_log_t worker_logs[MAX_TASKS];
int gpu_log_count = 0;
int worker_log_count = 0;
pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

// 결과 동기화를 위한 뮤텍스와 조건 변수
pthread_mutex_t result_mutex[MAX_GPU_QUEUE_SIZE];
pthread_cond_t result_cond[MAX_GPU_QUEUE_SIZE];

void print_layer_info(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL || l.type == CONNECTED){
            // printf("ConV/FC Layer %d: num_pseudo_layer=%d, pseudo_layer_indexes[num_pseudo_layer]=%d\n", i, num_pseudo_layer, pseudo_layer_indexes[num_pseudo_layer]);

            pseudo_layer_indexes[num_pseudo_layer] = i;
            num_pseudo_layer++;
        }
    }
    pseudo_layer_indexes[num_pseudo_layer] = net.n;
    num_pseudo_layer++;

    printf("Number of pseudo layer: %d, Last pseudo layer index: %d, Total (net.n): %d\n", num_pseudo_layer - 1, num_pseudo_layer, net.n);

    // 모든 pseudo 레이어 인덱스 출력 (디버깅)
    if (VISUAL) printf("Pseudo layer indexes (%d total): ", num_pseudo_layer);
    for (i = 0; i < num_pseudo_layer; i++) {
       if (VISUAL) printf("%d ", pseudo_layer_indexes[i]);
    }
   if (VISUAL) printf("\n");

}

// 시간 측정 함수
double current_time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// 로그 함수 - 배열에 저장
void save_gpu_log(gpu_task_t task) {
    pthread_mutex_lock(&log_mutex);
    if (gpu_log_count < MAX_TASKS) {
        gpu_logs[gpu_log_count].thread_id = task.thread_id;
        gpu_logs[gpu_log_count].core_id = task.core_id;
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

    fprintf(fp_gpu, "thread_id,core_id, Gstart,Gend,request_time,push_start_time,push_end_time,gpu_start_time,gpu_end_time,pull_start_time,pull_end_time,queue_waiting_delay,push_delay,gpu_inference_delay,pull_delay,total_delay\n");
    for (int i = 0; i < gpu_log_count; i++) {
        // 큐 대기 시간
        double queue_waiting_delay = gpu_logs[i].push_start_time - gpu_logs[i].request_time;
        double push_delay = gpu_logs[i].push_end_time - gpu_logs[i].push_start_time;
        // compute_delay를 gpu_inference_delay로 이름 변경
        double gpu_inference_delay = gpu_logs[i].gpu_end_time - gpu_logs[i].gpu_start_time;
        double pull_delay = gpu_logs[i].pull_end_time - gpu_logs[i].pull_start_time;
        double total_delay = gpu_logs[i].pull_end_time - gpu_logs[i].push_start_time;
        
        fprintf(fp_gpu, "%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                gpu_logs[i].thread_id,
                gpu_logs[i].core_id,
                gpu_logs[i].Gstart,
                gpu_logs[i].Gend,
                gpu_logs[i].request_time, 
                gpu_logs[i].push_start_time, 
                gpu_logs[i].push_end_time,
                gpu_logs[i].gpu_start_time, 
                gpu_logs[i].gpu_end_time,
                gpu_logs[i].pull_start_time,
                gpu_logs[i].pull_end_time,
                queue_waiting_delay,
                push_delay,
                gpu_inference_delay,
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

    // 워커 CSV 헤더 - GPU 지연 시간 필드의 위치 수정
    fprintf(fp_worker, "thread_id,core_id,Gstart,Gend,worker_start_time,worker_inference_time,worker_request_time,worker_receive_time,worker_postprocess_time,worker_end_time,preprocess_delay,cpu_inference_delay_1,queue_waiting_delay,push_delay,gpu_inference_delay,pull_delay,total_gpu_delay,cpu_inference_delay_2,postprocess_delay,total_delay\n");
    
    for (int i = 0; i < worker_log_count; i++) {
        // 워커 지연 시간 계산
        double preprocess_delay = worker_logs[i].worker_inference_time - worker_logs[i].worker_start_time;
        double cpu_inference_delay_1 = worker_logs[i].worker_request_time - worker_logs[i].worker_inference_time;
        double cpu_inference_delay_2 = worker_logs[i].worker_postprocess_time - worker_logs[i].worker_receive_time;
        double postprocess_delay = worker_logs[i].worker_end_time - worker_logs[i].worker_postprocess_time;
        double total_delay = worker_logs[i].worker_end_time - worker_logs[i].worker_start_time;
        
        // GPU 지연 시간 가져오기 (동일한 인덱스 i 사용)
        double queue_waiting_delay = 0.0;
        double push_delay = 0.0;
        double gpu_inference_delay = 0.0;
        double pull_delay = 0.0;
        double total_gpu_delay = 0.0;
        
        // GPU 로그와 워커 로그의 개수가 동일하다고 가정
        if (i < gpu_log_count) {
            queue_waiting_delay = gpu_logs[i].push_start_time - gpu_logs[i].request_time;
            push_delay = gpu_logs[i].push_end_time - gpu_logs[i].push_start_time;
            gpu_inference_delay = gpu_logs[i].gpu_end_time - gpu_logs[i].gpu_start_time;
            pull_delay = gpu_logs[i].pull_end_time - gpu_logs[i].pull_start_time;
            total_gpu_delay = gpu_logs[i].pull_end_time - gpu_logs[i].push_start_time;
        }
        
        // 워커 로그와 GPU 지연 시간 함께 저장 - GPU 관련 필드 위치 수정
        fprintf(fp_worker, "%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                worker_logs[i].thread_id,
                worker_logs[i].core_id,
                worker_logs[i].Gstart,
                worker_logs[i].Gend,
                worker_logs[i].worker_start_time, 
                worker_logs[i].worker_inference_time, 
                worker_logs[i].worker_request_time, 
                worker_logs[i].worker_receive_time, 
                worker_logs[i].worker_postprocess_time, 
                worker_logs[i].worker_end_time,
                preprocess_delay,
                cpu_inference_delay_1,
                // GPU 지연 시간 필드 (cpu_inference_delay_1과 cpu_inference_delay_2 사이로 이동)
                queue_waiting_delay,
                push_delay,
                gpu_inference_delay,
                pull_delay,
                total_gpu_delay,
                cpu_inference_delay_2,
                postprocess_delay,
                total_delay);
    }
    fclose(fp_worker);
}