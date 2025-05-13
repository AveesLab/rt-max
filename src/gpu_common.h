#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <stdlib.h>
#include "darknet.h"
#include "network.h"
#include "parser.h"
#include "option_list.h"
#include "detector.h"

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

#define MAX_SEGMENTS 100
#define START_IDX 3
#define VISUAL 0
#define MAX_BUFFER_SIZE 2097152

#define MAX_GPU_QUEUE_SIZE 128
#define MAX_TASKS 1000  // 최대 작업 수 (로그 저장용)

extern int coreIDOrder[MAXCORES];
extern int pseudo_layer_indexes[500];
extern int num_pseudo_layer;
extern int num_thread;
extern int num_exp;
extern int Gstart;
extern int Gend;
extern int gpu_index;
extern int gpu_yolo;

// GPU 작업 큐 관련 구조체
typedef struct gpu_task_t {
    float *input;              // 입력 데이터
    int size;                  // 입력 데이터 크기
    network net;               // 네트워크 정보
    int task_id;               // 작업 ID
    int thread_id;             // 요청한 스레드 ID
    int core_id;
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

    
    // 세그먼트 시간 (추가된 부분)
    double segment_times[MAX_SEGMENTS];
    int segment_starts[MAX_SEGMENTS];
    int segment_ends[MAX_SEGMENTS];
    int num_segments;

    // Skip connection을 위한 추가 필드
    int skip_count;                 // skip connection 개수
    int skip_layers_idx[10];        // skip connection 레이어 인덱스
    float *skip_layers_data[10];    // 각 skip connection 레이어의 데이터 포인터
    int skip_layers_size[10];       // 각 skip connection 레이어의 데이터 크기

    // 새로 추가된 필드
    char cfgfile[256]; // 모델 설정 파일 경로
} gpu_task_t;

// 로그 저장용 구조체
typedef struct gpu_log_t {
    int thread_id;
    int core_id;
    int Gstart;                // GPU 시작 레이어
    int Gend;                  // GPU 종료 레이어
    double request_time;
    double push_start_time;
    double push_end_time;
    double gpu_start_time;
    double gpu_end_time;
    double pull_start_time;
    double pull_end_time;

    // 세그먼트 시간 추가 (배열로 저장)
    double segment_times[MAX_SEGMENTS];
    int num_segments;
    int segment_starts[MAX_SEGMENTS];
    int segment_ends[MAX_SEGMENTS];
} gpu_log_t;

typedef struct worker_log_t {
    int thread_id;
    int core_id;
    int Gstart;                // GPU 시작 레이어
    int Gend;                  // GPU 종료 레이어
    double worker_start_time;
    double worker_inference_time;
    double worker_request_time;
    double worker_receive_time;
    double worker_postprocess_time;
    double worker_end_time;
    // 추가된 전송 시간 (워커 로그에도 GPU 작업 관련 시간 추가)
    double push_time;          // GPU 메모리로 전송 시간
    double compute_time;       // GPU 계산 시간
    double pull_time;          // GPU 메모리에서 전송 시간
} worker_log_t;

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

// 외부 변수 선언
extern pthread_barrier_t log_barrier;
extern pthread_mutex_t log_write_mutex;
extern int log_written;

extern FILE *fp_gpu;
extern FILE *fp_worker;

extern pthread_barrier_t barrier;
extern pthread_mutex_t mutex_init;

extern int skip_layers[1000][10];
extern pthread_mutex_t mutex_gpu;
extern pthread_cond_t cond;
extern int current_thread;

extern gpu_task_t gpu_task_queue[MAX_GPU_QUEUE_SIZE];
extern int gpu_task_head, gpu_task_tail;
extern pthread_mutex_t gpu_queue_mutex;
extern pthread_cond_t gpu_queue_cond;

extern gpu_log_t gpu_logs[MAX_TASKS];
extern worker_log_t worker_logs[MAX_TASKS];
extern int gpu_log_count;
extern int worker_log_count;
extern pthread_mutex_t log_mutex;

extern pthread_mutex_t result_mutex[MAX_GPU_QUEUE_SIZE];
extern pthread_cond_t result_cond[MAX_GPU_QUEUE_SIZE];

// 함수 선언
void print_layer_info(network net);
double current_time_in_ms();
void save_gpu_log(gpu_task_t task);
void save_worker_log(worker_log_t log);
int compare_gpu_logs(const void *a, const void *b);
int compare_worker_logs(const void *a, const void *b);
void write_logs_to_files(char *model_name, char *gpu_path, char *worker_path);

#endif // GPU_COMMON_H
