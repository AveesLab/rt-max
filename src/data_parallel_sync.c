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
#include <stdbool.h>

#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#define MAXCORES 8
#define MAX_EXP 1000

// CPU 코어 순서 정의 (예: 4~11번 코어 사용)
static int coreIDOrder[MAXCORES] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

// 스레드에 전달할 데이터 구조체
typedef struct thread_data_t {
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
    bool isTest;
} thread_data_t;

// 스레드 결과를 저장하기 위한 구조체
typedef struct {
    int thread_id;
    int num_iterations;
    double core_id[MAX_EXP];
    double e_preprocess[MAX_EXP];
    double e_infer[MAX_EXP];
    double e_postprocess[MAX_EXP];
    double execution_time[MAX_EXP];
} thread_result_t;

// 결과를 저장하기 위한 전역 배열
thread_result_t *thread_results;

// 동기화를 위한 barrier 및 mutex 변수 선언
pthread_barrier_t barrier;
pthread_mutex_t init_mutex;

// 스레드 함수
static void* threadFunc(void* arg)
{
    thread_data_t* data = (thread_data_t*)arg;

    int device = 0; // CPU 사용

    // __CPU AFFINITY SETTING__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreIDOrder[data->thread_id - 1], &cpuset); // CPU 코어 인덱스 설정
    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "pthread_setaffinity_np() failed for thread %d\n", data->thread_id);
        pthread_exit(NULL);
    }

    // 데이터 설정 로드
    list *options = read_data_cfg(data->datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size);

    char buff[256];
    char *input = buff;

    image **alphabet = load_alphabet();

    float nms = 0.45; // NMS 임계값

    int top = 5;

    char *target_model = "yolo";
    int object_detection = (strstr(data->cfgfile, target_model) != NULL);

    // 동일한 cfg, weights 파일 동시 접근을 막기 위한 lock
    pthread_mutex_lock(&init_mutex);

    // __각 스레드별로 네트워크를 로드__
    network net = parse_network_cfg_custom(data->cfgfile, 1, 1, device); // batch=1로 설정
    if (data->weightfile) {
        load_weights(&net, data->weightfile);
    }
    if (net.letter_box) data->letter_box = 1;
    net.benchmark_layers = data->benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    // lock 해제
    pthread_mutex_unlock(&init_mutex);

    layer l = net.layers[net.n - 1];

    srand(2222222 + data->thread_id); // 스레드별 시드 설정

    if (data->filename) strncpy(input, data->filename, 256);
    else {
        printf("Error! File does not exist.\n");
        pthread_exit(NULL);
    }

    // 스레드 결과 저장을 위한 포인터
    thread_result_t *result = &thread_results[data->thread_id - 1];
    result->thread_id = data->thread_id;
    result->num_iterations = num_exp;

    for (int i = 0; i < num_exp; i++) {
        // __각 반복(iteration)의 시작에서 동기화__
        pthread_barrier_wait(&barrier);

        // __Preprocess (이미지 전처리)__
        double start_time = get_time_in_ms();
        image im = load_image(input, 0, 0, net.c);
        image resized = resize_min(im, net.w);
        image cropped = crop_image(resized, (resized.w - net.w) / 2, (resized.h - net.h) / 2, net.w, net.h);
        float *X = cropped.data;
        result->e_preprocess[i] = get_time_in_ms() - start_time;

        // __Inference (CPU 추론)__
        start_time = get_time_in_ms();
        float *predictions = network_predict_cpu(net, X);
        result->e_infer[i] = get_time_in_ms() - start_time;

        // __Postprocess (후처리)__
        start_time = get_time_in_ms();
        if (object_detection) {
            int nboxes = 0;
            detection *dets = get_network_boxes(&net, im.w, im.h, data->thresh, data->hier_thresh, 0, 1, &nboxes, data->letter_box);
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS)
                    do_nms_sort(dets, nboxes, l.classes, nms);
                else
                    diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }
            // 결과를 이미지에 그리거나 처리
            // draw_detections_v3(im, dets, nboxes, data->thresh, names, alphabet, l.classes, data->ext_output);
            free_detections(dets, nboxes); // 메모리 해제
        } else {
            int *indexes = (int *)xcalloc(top, sizeof(int));
            if (net.hierarchy)
                hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
            top_k(predictions, net.outputs, top, indexes);
            // 결과 출력 또는 처리
            free(indexes); // 메모리 해제
        }
        result->e_postprocess[i] = get_time_in_ms() - start_time;
        result->execution_time[i] = result->e_preprocess[i] + result->e_infer[i] + result->e_postprocess[i];
        result->core_id[i] = (double)sched_getcpu();

        // 메모리 해제
        free_image(im);
        free_image(resized);
        free_image(cropped);
    }

    // 메모리 해제
    free_ptrs((void **)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_alphabet(alphabet);
    // free_network(net); // 필요한 경우 해제

    pthread_exit(NULL);
}

// 데이터 병렬 처리 함수
void data_parallel_sync(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
                   float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    pthread_t threads[num_thread];
    int rc;
    thread_data_t data[num_thread];

    printf("\n\nData-parallel\n");
    printf("\n::TEST:: Data-parallel with %d threads\n", num_thread);

    // 결과를 저장할 메모리 할당
    thread_results = (thread_result_t *)calloc(num_thread, sizeof(thread_result_t));
    if (!thread_results) {
        fprintf(stderr, "Error allocating memory for thread_results\n");
        exit(-1);
    }

    // 동기화를 위한 barrier 및 mutex 초기화
    if (pthread_barrier_init(&barrier, NULL, num_thread) != 0) {
        fprintf(stderr, "Could not initialize barrier\n");
        exit(-1);
    }

    if (pthread_mutex_init(&init_mutex, NULL) != 0) {
        fprintf(stderr, "Could not initialize mutex\n");
        exit(-1);
    }

    // __스레드별 데이터 설정 및 생성__
    for (int i = 0; i < num_thread; i++) {
        data[i].datacfg = datacfg;
        data[i].cfgfile = cfgfile;
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
        data[i].isTest = true;

        rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
        if (rc) {
            printf("Error: Unable to create thread %d, %d\n", i, rc);
            exit(-1);
        }
    }

    // 스레드 종료 대기
    for (int i = 0; i < num_thread; i++) {
        pthread_join(threads[i], NULL);
    }

    // barrier 및 mutex 해제
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&init_mutex);

    // 결과 출력
    for (int i = 0; i < num_thread; i++) {
        thread_result_t *result = &thread_results[i];
        for (int j = 0; j < result->num_iterations; j++) {
            printf("Thread %d, Iteration %d --- [Core ID: %0.0lf] Pre: %.3lf ms, Infer: %.3lf ms, Post: %.3lf ms, Total: %.3lf ms\n",
                   result->thread_id, j, result->core_id[j], result->e_preprocess[j], result->e_infer[j], result->e_postprocess[j], result->execution_time[j]);
        }
    }

    // __추가: Iteration 30 이상 40 이하의 각 단계 평균 지연 시간 계산 및 출력__
    int start_iter = 30;
    int end_iter = 40;
    int count = 0;
    double sum_pre = 0.0, sum_infer = 0.0, sum_post = 0.0, sum_total = 0.0;

    for (int i = 0; i < num_thread; i++) {
        thread_result_t *result = &thread_results[i];
        for (int j = start_iter; j <= end_iter && j < result->num_iterations; j++) {
            sum_pre += result->e_preprocess[j];
            sum_infer += result->e_infer[j];
            sum_post += result->e_postprocess[j];
            sum_total += result->execution_time[j];
            count++;
        }
    }

    if (count > 0) {
        double avg_pre = sum_pre / count;
        double avg_infer = sum_infer / count;
        double avg_post = sum_post / count;
        double avg_total = sum_total / count;

        printf("\n=== Average Delays for Iterations %d to %d ===\n", start_iter, end_iter);
        printf("Preprocess: %.3lf ms, Infer: %.3lf ms, Postprocess: %.3lf ms, Total: %.3lf ms\n",
               avg_pre, avg_infer, avg_post, avg_total);
    } else {
        printf("\nNo iterations found in the range %d to %d.\n", start_iter, end_iter);
    }

    // 메모리 해제
    free(thread_results);
}
