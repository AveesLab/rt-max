#include <stdlib.h>
#include <math.h>
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

#define START_INDEX 5
#define END_INDEX 2
#define ACCEPTABLE_JITTER 3

pthread_barrier_t barrier;
pthread_barrier_t barrier_reclaiming;
static int coreIDOrder[MAXCORES] = {0, 3, 6, 9, 4, 7, 10, 2, 5, 8, 11, 1};
// static int coreIDOrder[MAXCORES] = {0,2,3,4,5,6,7,8,9,10,11,1};
static network net_list[MAXCORES];
static pthread_mutex_t mutex_init = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t mutex_post = PTHREAD_MUTEX_INITIALIZER;
static double start_time[MAXCORES] = {0,};
static int openblas_thread;
static pthread_mutex_t mutex_gpu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t mutex_reclaim = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static int current_thread = 1;
static double execution_time_wo_waiting;

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
    bool isSet;
    bool isReclaiming;
} thread_data_t;

#ifdef MEASURE
static double core_id_list[1000];
static double start_preprocess[1000];
static double end_preprocess[1000];
static double e_preprocess[1000];
static double e_preprocess_max[1000];

static double start_infer[1000];
static double start_gpu_waiting[1000];
static double start_gpu_infer[1000];
static double end_gpu_infer[1000];
static double start_reclaim_waiting[1000];
static double start_reclaim_infer[1000];
static double end_reclaim_infer[1000];
static double start_cpu_infer[1000];
static double end_cpu_infer[1000];
static double end_infer[1000];

static double waiting_gpu[1000];
static double e_gpu_infer[1000];
static double e_gpu_infer_max[1000];

static double waiting_reclaim[1000];
static double e_reclaim_infer[1000];
static double e_cpu_infer[1000];
static double e_infer[1000];


static double start_postprocess[1000];
static double end_postprocess[1000];
static double e_postprocess[1000];

static double check_jitter[1000] = {0, };
static int optimal_core;
static float R;
#endif

static double execution_time[1000];
static double execution_time_max[1000];

static double frame_rate[1000];
static float avg_preprocess_time;
static float avg_gpu_infer_time;
static float avg_execution_time;

static float max_preprocess_time;
static float max_gpu_infer_time;
static float max_execution_time;
static float sleep_time;
static float R;

static int reset_check_jitter() {
    for (int check_num = 0; check_num < 1000; check_num++) {
        check_jitter[check_num] = 0;
    }
    return 1;
}

static int is_GPU_larger(double a, double b) {
    return (a - b) >= 2 ? 1 : 0; // Check 2ms differnce
}

static double average(double arr[]){
    double sum;
    int total_num_exp = num_exp * optimal_core;
    int skip_num_exp =  START_INDEX * optimal_core;
    int end_num_exp =  END_INDEX * optimal_core;
    int i;
    for(i = skip_num_exp ; i < total_num_exp - end_num_exp; i++) {
        sum += arr[i];
    }
    return (sum / (total_num_exp-skip_num_exp-end_num_exp)) * 1.03;
}

#ifdef MEASURE

static int compare(const void *a, const void *b) {
    double valueA = *((double *)a + 1);
    double valueB = *((double *)b + 1);

    if (valueA < valueB) return -1;
    if (valueA > valueB) return 1;
    return 0;
}

static double maxOfThree(double a, double b, double c) {
    double max = a;

    if (b > max) {
        max = b;
    }
    if (c > max) { 
        max = c;
    }
    return max; 
}

static int write_result_gpu(char *file_path) 
{
    static int exist=0;
    FILE *fp;
    int tick = 0;

    fp = fopen(file_path, "w+");

    int i;
    if (fp == NULL) 
    {
        /* make directory */
        while(!exist)
        {
            int result;

            usleep(10 * 1000);

            result = mkdir(MEASUREMENT_PATH, 0766);
            if(result == 0) { 
                exist = 1;

                fp = fopen(file_path,"w+");
            }

            if(tick == 100)
            {
                fprintf(stderr, "\nERROR: Fail to Create %s\n", file_path);

                return -1;
            }
            else tick++;
        }
    }
    else printf("Write output in %s\n", file_path); 

    double sum_measure_data[num_exp * optimal_core][30];
    for(i = 0; i < num_exp * optimal_core; i++)
    {
        sum_measure_data[i][0] = core_id_list[i];
        sum_measure_data[i][1] = start_preprocess[i];
        sum_measure_data[i][2] = e_preprocess[i];
        sum_measure_data[i][3] = end_preprocess[i];
        sum_measure_data[i][4] = e_preprocess_max[i];
        sum_measure_data[i][5] = max_preprocess_time;
        sum_measure_data[i][6] = start_infer[i]; 
        sum_measure_data[i][7] = start_cpu_infer[i];
        sum_measure_data[i][8] = e_cpu_infer[i];
        sum_measure_data[i][9] = end_cpu_infer[i];
        sum_measure_data[i][10] = start_gpu_waiting[i];
        sum_measure_data[i][11] = waiting_gpu[i];
        sum_measure_data[i][12] = start_gpu_infer[i];
        sum_measure_data[i][13] = e_gpu_infer[i];
        sum_measure_data[i][14] = end_gpu_infer[i];
        sum_measure_data[i][15] = e_gpu_infer_max[i];
        sum_measure_data[i][16] = max_gpu_infer_time;
        sum_measure_data[i][17] = end_infer[i];
        sum_measure_data[i][18] = e_infer[i];
        sum_measure_data[i][19] = start_postprocess[i];
        sum_measure_data[i][20] = e_postprocess[i];
        sum_measure_data[i][21] = end_postprocess[i];
        sum_measure_data[i][22] = execution_time[i];
        sum_measure_data[i][23] = execution_time_max[i];
        sum_measure_data[i][24] = max_execution_time;
        sum_measure_data[i][25] = 0.0;
        sum_measure_data[i][26] = 0.0;
        sum_measure_data[i][27] = 0.0;        
        sum_measure_data[i][28] = 0.0;
        sum_measure_data[i][29] = check_jitter[i];
    }

    qsort(sum_measure_data, sizeof(sum_measure_data)/sizeof(sum_measure_data[0]), sizeof(sum_measure_data[0]), compare);

    int startIdx = optimal_core * START_INDEX; // Delete some ROWs
    int endIdx = optimal_core * END_INDEX; // Delete some ROWs
    double new_sum_measure_data[sizeof(sum_measure_data)/sizeof(sum_measure_data[0])-startIdx-endIdx][sizeof(sum_measure_data[0])];

    int newIndex = 0;
    for (int i = startIdx; i < sizeof(sum_measure_data)/sizeof(sum_measure_data[0])-endIdx; i++) {
        for (int j = 0; j < sizeof(sum_measure_data[0]); j++) {
            new_sum_measure_data[newIndex][j] = sum_measure_data[i][j];
        }
        newIndex++;
    }

        fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
            "core_id", 
            "start_preprocess", "e_preprocess", "end_preprocess", "e_preprocess_max", "e_preprocess_max_value",
            "start_infer", "start_cpu_infer", "e_cpu_infer", "end_cpu_infer",
            "start_gpu_waiting", "waiting_gpu", 
            "start_gpu_infer", "e_gpu_infer", "end_gpu_infer", "e_gpu_infer_max", "e_gpu_infer_max_value",
            "end_infer", "e_infer",
            "start_postprocess", "e_postprocess", "end_postprocess", 
            "execution_time", "execution_time_max", "execution_time_max_value",
            "cycle_time", "frame_rate",
            "optimal_core", "R", "check_jitter");

    double frame_rate = 0.0;
    double cycle_time = 0.0;

    for(i = 0; i < num_exp * optimal_core - startIdx - endIdx; i++)
    {
        if (new_sum_measure_data[i][29] > ACCEPTABLE_JITTER) continue;
        
        if (i == 0) cycle_time = NAN;
        else cycle_time = new_sum_measure_data[i][1] - new_sum_measure_data[i-1][1];

        if (i == 0) frame_rate = NAN;
        else frame_rate = 1000/cycle_time;

        new_sum_measure_data[i][25] = cycle_time;
        new_sum_measure_data[i][26] = frame_rate;
        new_sum_measure_data[i][27] = (double)optimal_core;
        new_sum_measure_data[i][28] = R;

        fprintf(fp, "%0.0f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f, %0.0f, %0.2f,%0.2f\n",  
                new_sum_measure_data[i][0], new_sum_measure_data[i][1], new_sum_measure_data[i][2], new_sum_measure_data[i][3], 
                new_sum_measure_data[i][4], new_sum_measure_data[i][5], new_sum_measure_data[i][6], new_sum_measure_data[i][7], 
                new_sum_measure_data[i][8], new_sum_measure_data[i][9], new_sum_measure_data[i][10], new_sum_measure_data[i][11], 
                new_sum_measure_data[i][12], new_sum_measure_data[i][13], new_sum_measure_data[i][14], new_sum_measure_data[i][15],
                new_sum_measure_data[i][16], new_sum_measure_data[i][17], new_sum_measure_data[i][18], new_sum_measure_data[i][19],
                new_sum_measure_data[i][20], new_sum_measure_data[i][21], new_sum_measure_data[i][22], new_sum_measure_data[i][23], 
                new_sum_measure_data[i][24], new_sum_measure_data[i][25], new_sum_measure_data[i][26], new_sum_measure_data[i][27],
                new_sum_measure_data[i][28], new_sum_measure_data[i][29]);
    }
    
    fclose(fp);

    return 1;
}


static int write_result_reclaiming(char *file_path) 
{
    static int exist=0;
    FILE *fp;
    int tick = 0;

    fp = fopen(file_path, "w+");

    int i;
    if (fp == NULL) 
    {
        /* make directory */
        while(!exist)
        {
            int result;

            usleep(10 * 1000);

            result = mkdir(MEASUREMENT_PATH, 0766);
            if(result == 0) { 
                exist = 1;

                fp = fopen(file_path,"w+");
            }

            if(tick == 100)
            {
                fprintf(stderr, "\nERROR: Fail to Create %s\n", file_path);

                return -1;
            }
            else tick++;
        }
    }
    else printf("Write output in %s\n", file_path); 

    double sum_measure_data[num_exp * optimal_core][30];
    for(i = 0; i < num_exp * optimal_core; i++)
    {
        sum_measure_data[i][0] = core_id_list[i];
        // printf("%d == %.0lf %.0lf\n", i, core_id_list[i], sum_measure_data[i][0]);
        sum_measure_data[i][1] = start_preprocess[i];     
        sum_measure_data[i][2] = e_preprocess[i];       
        sum_measure_data[i][3] = end_preprocess[i];
        sum_measure_data[i][4] = start_infer[i];
        sum_measure_data[i][5] = start_cpu_infer[i];     
        sum_measure_data[i][6] = e_cpu_infer[i];  
        sum_measure_data[i][7] = end_cpu_infer[i];  
        sum_measure_data[i][8] = waiting_reclaim[i];
        sum_measure_data[i][9] = start_reclaim_infer[i];    
        sum_measure_data[i][10] = e_reclaim_infer[i];    
        sum_measure_data[i][11] = end_reclaim_infer[i];
        sum_measure_data[i][12] = start_gpu_waiting[i];    
        sum_measure_data[i][13] = waiting_gpu[i];
        sum_measure_data[i][14] = start_gpu_infer[i];       
        sum_measure_data[i][15] = e_gpu_infer[i];        
        sum_measure_data[i][16] = end_gpu_infer[i];
        sum_measure_data[i][17] = end_infer[i];
        sum_measure_data[i][18] = e_infer[i];
        sum_measure_data[i][19] = start_postprocess[i];     
        sum_measure_data[i][20] = e_postprocess[i];      
        sum_measure_data[i][21] = end_postprocess[i];
        sum_measure_data[i][22] = execution_time[i];     
        sum_measure_data[i][23] = execution_time_max[i];
        sum_measure_data[i][24] = max_execution_time;     
        sum_measure_data[i][25] = 0.0;      
        sum_measure_data[i][26] = 0.0;
        sum_measure_data[i][27] = 0.0;
        sum_measure_data[i][28] = 0.0;
        sum_measure_data[i][29] = check_jitter[i];
    }

    qsort(sum_measure_data, sizeof(sum_measure_data)/sizeof(sum_measure_data[0]), sizeof(sum_measure_data[0]), compare);

    int startIdx = optimal_core * START_INDEX; // Delete some ROWs
    int endIdx = optimal_core * END_INDEX; // Delete some ROWs
    double new_sum_measure_data[sizeof(sum_measure_data)/sizeof(sum_measure_data[0])-startIdx-endIdx][sizeof(sum_measure_data[0])];

    int newIndex = 0;
    for (int i = startIdx; i < sizeof(sum_measure_data)/sizeof(sum_measure_data[0])-endIdx; i++) {
        for (int j = 0; j < sizeof(sum_measure_data[0]); j++) {
            new_sum_measure_data[newIndex][j] = sum_measure_data[i][j];
        }
        newIndex++;
    }

    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
            "core_id", 
            "start_preprocess", "e_preprocess", "end_preprocess", 
            "start_infer", "start_cpu_infer", "e_cpu_infer", "end_cpu_infer",
            "waiting_reclaim", "start_reclaim_infer", "e_reclaim_infer", "end_reclaim_infer", 
            "start_gpu_waiting", "waiting_gpu", 
            "start_gpu_infer", "e_gpu_infer", "end_gpu_infer",
            "end_infer", 
            "e_infer",
            "start_postprocess", "e_postprocess", "end_postprocess", 
            "execution_time", "execution_time_max", "execution_time_max_value",
            "cycle_time", "frame_rate",
            "optimal_core","R", "check_jitter");

    double frame_rate = 0.0;
    double cycle_time = 0.0;

    for(i = 0; i < num_exp * optimal_core - startIdx-endIdx; i++)
    {
        if (new_sum_measure_data[i][29] > ACCEPTABLE_JITTER) continue;
        
        if (i == 0) cycle_time = NAN;
        else cycle_time = new_sum_measure_data[i][1] - new_sum_measure_data[i-1][1];

        if (i == 0) frame_rate = NAN;
        else frame_rate = 1000/cycle_time;

        new_sum_measure_data[i][25] = cycle_time;
        new_sum_measure_data[i][26] = frame_rate;
        new_sum_measure_data[i][27] = (double)optimal_core;
        new_sum_measure_data[i][28] = R;

        fprintf(fp, "%0.0f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.0f,%0.2f,%0.2f\n",  
                new_sum_measure_data[i][0], new_sum_measure_data[i][1], new_sum_measure_data[i][2], 
                new_sum_measure_data[i][3], new_sum_measure_data[i][4], new_sum_measure_data[i][5], 
                new_sum_measure_data[i][6], new_sum_measure_data[i][7], new_sum_measure_data[i][8], 
                new_sum_measure_data[i][9], new_sum_measure_data[i][10], new_sum_measure_data[i][11],
                new_sum_measure_data[i][12], new_sum_measure_data[i][13], new_sum_measure_data[i][14], 
                new_sum_measure_data[i][15], new_sum_measure_data[i][16], new_sum_measure_data[i][17], 
                new_sum_measure_data[i][18], new_sum_measure_data[i][19], new_sum_measure_data[i][20], 
                new_sum_measure_data[i][21], new_sum_measure_data[i][22], new_sum_measure_data[i][23],
                new_sum_measure_data[i][24], new_sum_measure_data[i][25], new_sum_measure_data[i][26], 
                new_sum_measure_data[i][27], new_sum_measure_data[i][28], new_sum_measure_data[i][29]);
    }
    
    fclose(fp);

    return 1;
}
#endif

#ifdef GPU
static void threadFunc(thread_data_t data)
{
    // __CPU AFFINITY SETTING__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreIDOrder[data.thread_id], &cpuset); // cpu core index
    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "pthread_setaffinity_np() failed \n");
        exit(0);
    } 

    // __GPU SETUP__
#ifdef GPU   // GPU
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
        CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    }
#ifdef CUDNN_HALF
    printf(" CUDNN_HALF=1 \n");
#endif  // CUDNN_HALF
#else
    gpu_index = -1;
    printf(" GPU isn't used \n");
    init_cpu();
#endif  // GPU

    list *options = read_data_cfg(data.datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list)

    char buff[256];
    char *input = buff;

    image **alphabet = load_alphabet();

    float nms = .45;    // 0.4F
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

    int device = 1; // Choose CPU or GPU
    extern gpu_yolo;

    pthread_mutex_lock(&mutex_init);
    // double start_1 = get_time_in_ms();
    if (!data.isSet) {
        network net_init = parse_network_cfg_custom(data.cfgfile, 1, 1, device); // set batch=1

        if (data.weightfile) {
            load_weights(&net_init, data.weightfile);
        }
        if (net_init.letter_box) data.letter_box = 1;
        net_init.benchmark_layers = data.benchmark_layers;
        fuse_conv_batchnorm(net_init);
        calculate_binary_weights(net_init);

        net_list[data.thread_id] = net_init;
    }
    network net = net_list[data.thread_id];
    // network net = parse_network_cfg_custom(data.cfgfile, 1, 1, device);
    // printf("parse_network_cfg_custom : %.3lf ms\n", get_time_in_ms() - start_1);
    pthread_mutex_unlock(&mutex_init);

    layer l = net.layers[net.n - 1];

    extern int skip_layers[1000][10];
    int skipped_layers[1000] = {0, };

    for(i = gLayer; i < net.n; i++) {
        for(j = 0; j < 10; j++) {
            if((skip_layers[i][j] < gLayer)&&(skip_layers[i][j] != 0)) {
                skipped_layers[skip_layers[i][j]] = 1;
                // printf("skip layer[%d][%d] : %d,  \n", i, j, skip_layers[i][j]);
            }
        }
    }
    srand(2222222);
    double remaining_time = 0.0;

    if (data.filename) strncpy(input, data.filename, 256);
    else printf("Error! File is not exist.");
    openblas_thread = (MAXCORES - 2) - data.num_thread + 1;
    openblas_set_num_threads(openblas_thread);

    for (int k = 0; k < openblas_thread - 1; k++) {
        CPU_ZERO(&cpuset);
        CPU_SET(coreIDOrder[(MAXCORES - 2) - k], &cpuset);
        // printf("Rcore : %d\n",coreIDOrder[(MAXCORES - 2) - k] );
        openblas_setaffinity(k, sizeof(cpuset), &cpuset);
    }
    for (i = 0; i < num_exp; i++) {

#ifdef MEASURE
        int count = i * data.num_thread + data.thread_id - 1;
#endif

        if(!data.isTest) {
            if(i < START_INDEX) {
            	if (data.isReclaiming){
                    pthread_barrier_wait(&barrier_reclaiming);
                    usleep(R * (data.thread_id - 1) * 1000);
            	}
            	else {
                    pthread_barrier_wait(&barrier);
                    usleep(R * (data.thread_id - 1) * 1000);
            	}
            	start_time[data.thread_id]=get_time_in_ms();
            }
            else{
                start_time[data.thread_id] +=  R * optimal_core;
                remaining_time = start_time[data.thread_id] - get_time_in_ms();
                if (remaining_time > 0) usleep(remaining_time * 1000);
                else if (remaining_time < -ACCEPTABLE_JITTER) {
                    for (int idx = 1; idx < optimal_core + 2; idx++) {
			start_time[idx] += (fabs(remaining_time) * 2);
			check_jitter[count + idx - optimal_core] = fabs(remaining_time);
			if (idx == 1) check_jitter[count + idx - optimal_core -1] = fabs(remaining_time);
		    }
		}
            }
        }


#ifdef NVTX
        char task[100];
        sprintf(task, "Task (cpu: %d)", data.thread_id);
        nvtxRangeId_t nvtx_task;
        nvtx_task = nvtxRangeStartA(task);
#endif

#ifdef MEASURE
        //printf("\nThread %d is set to CPU core %d count(%d) : %d \n\n", data.thread_id, sched_getcpu(), data.thread_id, count);
#else
        printf("\nThread %d is set to CPU core %d\n\n", data.thread_id, sched_getcpu());
#endif

        time = get_time_in_ms();
        // __Preprocess__
#ifdef MEASURE
        start_preprocess[count] = get_time_in_ms();
#endif
        im = load_image(input, 0, 0, net.c);
        resized = resize_min(im, net.w);
        cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        X = cropped.data;

#ifdef MEASURE
        end_preprocess[count] = get_time_in_ms();
        e_preprocess[count] = end_preprocess[count] - start_preprocess[count];
#endif
        e_preprocess_max[count] = get_time_in_ms() - start_preprocess[count];

        // __Inference__
        // if (device) predictions = network_predict(net, X);
        // else predictions = network_predict_cpu(net, X);

#ifdef MEASURE
        start_infer[count] = get_time_in_ms();
#endif

        network_state state;
        state.index = 0;
        state.net = net;
        state.input = X;
        // state.input = net.input_state_gpu;
        // memcpy(net.input_pinned_cpu, X, size * sizeof(float));
        state.truth = 0;
        state.train = 0;
        state.delta = 0;

        // CPU Inference
#ifdef NVTX
        char task_cpu[100];
        sprintf(task_cpu, "Task (cpu: %d) - CPU Inference", data.thread_id);
        nvtxRangeId_t nvtx_task_cpu;
        nvtx_task_cpu = nvtxRangeStartA(task_cpu);
#endif

#ifdef MEASURE
        start_cpu_infer[count] = get_time_in_ms();
#endif

        state.workspace = net.workspace_cpu;
        gpu_yolo = 0;

        int end_layer_cpu = 0;
        if (data.isReclaiming) {
            end_layer_cpu = rLayer;
        }
        else {
            end_layer_cpu = gLayer;
        }

        for(j = 0; j < end_layer_cpu; ++j){
            // printf("%d %d\n", j, sched_getcpu());
            state.index = j;
            l = net.layers[j];
            l.do_reclaiming = 0;
            if(l.delta && state.train && l.train){
                scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
            }
            l.forward(l, state);
            if (skipped_layers[j]){
                cuda_push_array(l.output_gpu, l.output, l.outputs * l.batch);
            }
            state.input = l.output;
        }

#ifdef MEASURE
        end_cpu_infer[count] = get_time_in_ms();
#endif

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_cpu);
#endif

        // Reclaiming Inference
        if (data.isReclaiming) {

#ifdef NVTX
            char task_reclaiming[100];
            sprintf(task_reclaiming, "Task (cpu: %d) - Reclaiming Inference", data.thread_id);
            nvtxRangeId_t nvtx_task_reclaiming;
            nvtx_task_reclaiming = nvtxRangeStartA(task_reclaiming);
#endif

#ifdef MEASURE
            start_reclaim_waiting[count] = get_time_in_ms();
#endif

            pthread_mutex_lock(&mutex_reclaim);

#ifdef MEASURE
            start_reclaim_infer[count] = get_time_in_ms();
#endif



            for(j = rLayer; j < gLayer; ++j){
                state.index = j;
                l = net.layers[j];
                l.do_reclaiming = 1;
                if(l.delta && state.train && l.train){
                    scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
                }
                l.forward(l, state);
                if (skip_layers[j]) {
                    cuda_push_array(l.output_gpu, l.output, l.outputs * l.batch);
                }
                state.input = l.output;
            }

            pthread_mutex_unlock(&mutex_reclaim);

#ifdef MEASURE
        end_reclaim_infer[count] = get_time_in_ms();
#endif

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_reclaiming);
#endif
        }
        else{
            start_reclaim_waiting[count] = 0;
            start_reclaim_infer[count] = 0;
            end_reclaim_infer[count] = 0;

        }

        // GPU Inference

#ifdef NVTX
        char task_gpu[100];
        sprintf(task_gpu, "Task (cpu: %d) - GPU Inference", data.thread_id);
        nvtxRangeId_t nvtx_task_gpu;
        nvtx_task_gpu = nvtxRangeStartA(task_gpu);
#endif

#ifdef MEASURE
        start_gpu_waiting[count] = get_time_in_ms();
#endif

        pthread_mutex_lock(&mutex_gpu);

#ifdef MEASURE
        start_gpu_infer[count] = get_time_in_ms();
#endif
        if (net.gpu_index != cuda_get_device())
            cuda_set_device(net.gpu_index);

        cuda_push_array(l.output_gpu, l.output, l.outputs * l.batch);
        state.input = l.output_gpu;

        state.workspace = net.workspace;

        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

        for(j = gLayer; j < net.n; ++j){
            state.index = j;
            l = net.layers[j];
            if(l.delta_gpu && state.train){
                fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
            }

            l.forward_gpu(l, state);
            state.input = l.output_gpu;
        }

        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

        if (gLayer == net.n) predictions = get_network_output(net, 0);
        else predictions = get_network_output_gpu(net);
        reset_wait_stream_events();
        //cuda_free(state.input);   // will be freed in the free_network()

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_gpu);
#endif

#ifdef MEASURE
        end_gpu_infer[count] = get_time_in_ms();
#endif
        e_gpu_infer_max[count] = get_time_in_ms() - start_gpu_waiting[count]; // [+] Waiting_GPU Time

        // if (data.thread_id == data.num_thread) {
        //     current_thread = 1;
        // } else {
        //     current_thread++;
        // }

        // pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mutex_gpu);

#ifdef MEASURE
        end_infer[count] = get_time_in_ms();
        e_cpu_infer[count] = end_cpu_infer[count] - start_cpu_infer[count];
        waiting_reclaim[count] = start_reclaim_infer[count] - end_cpu_infer[count];//start_reclaim_waiting[i];
        e_reclaim_infer[count] = end_reclaim_infer[count] - start_reclaim_infer[count];
        waiting_gpu[count] = start_gpu_infer[count] - start_gpu_waiting[count];
        e_gpu_infer[count] = end_gpu_infer[count] - start_gpu_infer[count];
        e_infer[count] = end_infer[count] - start_infer[count];
        // printf("gpu : %0.2f, reclaim : %0.2f, cpu : %0.2f \n", e_gpu_infer[count], e_reclaim_infer[count], e_cpu_infer[count]);
#endif
        // printf("%0.2f %0.2f %0.2f\n",end_gpu_infer[count], waiting_reclaim[count], start_reclaim_infer[count]);
        // __Postprecess__
#ifdef MEASURE
        start_postprocess[count] = get_time_in_ms();
#endif

        // pthread_mutex_lock(&mutex_post);

        // __NMS & TOP acccuracy__
        if (object_detection) {
            dets = get_network_boxes(&net, im.w, im.h, data.thresh, data.hier_thresh, 0, 1, &nboxes, data.letter_box);
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms); 
                else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }
            // draw_detections_v3(im, dets, nboxes, data.thresh, names, alphabet, l.classes, data.ext_output);
        }
        else {
            if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
            top_k(predictions, net.outputs, top, indexes);
            for(j = 0; j < top; ++j){
                index = indexes[j];
                if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
                 else if (data.thread_id == 1 && i == 3)printf("%s: %f\n",names[index], predictions[index]);

            }
        }
        // pthread_mutex_unlock(&mutex_post);

        // __Display__
        // if (!data.dont_show) {
        //     show_image(im, "predictions");
        //     wait_key_cv(1);
        // }

        // free memory
        free_image(im);
        free_image(resized);
        free_image(cropped);

#ifdef MEASURE
        end_postprocess[count] = get_time_in_ms();
        e_postprocess[count] = end_postprocess[count] - start_postprocess[count];
        execution_time[count] = end_postprocess[count] - start_preprocess[count];
        core_id_list[count] = (double)sched_getcpu();
        // printf("%d -- %.0f\n", count, core_id_list[count]);
        // printf("\n%s: Predicted in %0.3f milli-seconds.\n", input, e_infer[count]);
#else
        execution_time[i] = get_time_in_ms() - time;
        frame_rate[i] = 1000.0 / (execution_time[i] / data.num_thread); // N thread
        printf("\n%s: Predicted in %0.3f milli-seconds. (%0.3lf fps)\n", input, execution_time[i], frame_rate[i]);
#endif
        // if (!data.isTest) {
        //     remaining_time = R *optimal_core - (get_time_in_ms() - start_preprocess[count]);
        //     if (remaining_time > 0) usleep(remaining_time * 1000);
        // }

        execution_time_max[count] = get_time_in_ms() - start_preprocess[count];

#ifdef NVTX
        nvtxRangeEnd(nvtx_task);
#endif
    }

    // free memory
    free_detections(dets, nboxes);
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_alphabet(alphabet);
    // free_network(net);

    pthread_exit(NULL);

}


void cpu_reclaiming_CRG(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    num_thread = MAXCORES - 2;
    bool visible_exp = false;
    visible_exp = true;
    
    if (visible_exp) printf("\nCPU-Reclaiming with %d threads with %d gpu-layer & %d reclaim-layer\n", num_thread, gLayer, rLayer);

    pthread_t threads[MAXCORES - 2];
    int rc;
    int i;

    thread_data_t data[MAXCORES - 2];

    if (opt_core == 0) {
        optimal_core = 10;

        R = 0.0;
        sleep_time = 0.0;
        max_preprocess_time = 0.0;
        max_gpu_infer_time = 0.0;
        max_execution_time = 0.0;
        avg_preprocess_time = 0.0;
        avg_gpu_infer_time = 0.0;
        avg_execution_time = 0.0;

        // printf("\n\nGPU-accelerated with Jitter Compensation (CS: \"GPU\")\n");
        if (visible_exp) printf("\n::TEST:: GPU-Accel with %d threads with %d gpu-layer\n", optimal_core, gLayer);

        for (i = 0; i < optimal_core; i++) {
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
            data[i].num_thread = optimal_core;
            data[i].isTest = true;
            data[i].isSet = false;
            data[i].isReclaiming = false;
            rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
            if (rc) {
                printf("Error: Unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        for (i = 0; i < optimal_core; i++) {
            pthread_join(threads[i], NULL);
        }


        execution_time_wo_waiting = (average(e_preprocess)+average(e_cpu_infer)+average(e_gpu_infer)+average(e_postprocess));

        if (visible_exp) {
            printf("e_pre : %0.02f, e_infer_cpu : %0.02f, e_infer_gpu : %0.02f, CPU/N: %0.02f\n", average(e_preprocess), average(e_cpu_infer), average(e_gpu_infer), execution_time_wo_waiting/(MAXCORES - 2));
        }

        R = MAX(average(e_gpu_infer), execution_time_wo_waiting/(MAXCORES - 2));
        optimal_core = (int)ceil(execution_time_wo_waiting / R);
        if (optimal_core > (MAXCORES - 2)) optimal_core = MAXCORES - 2;
        max_execution_time = R * optimal_core;
        
        if (visible_exp) printf("\n::EXP-1:: GPU-Accel with %d threads with %d gpu-layer\n", optimal_core, gLayer);
        reset_check_jitter();
        pthread_barrier_init(&barrier, NULL, optimal_core);
        for (i = 0; i < optimal_core; i++) {
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
            data[i].num_thread = optimal_core;
            data[i].isTest = false;
            data[i].isSet = true;
            data[i].isReclaiming = false;
            rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
            if (rc) {
                printf("Error: Unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        for (i = 0; i < optimal_core; i++) {
            pthread_join(threads[i], NULL);
        }


        execution_time_wo_waiting = (average(e_preprocess)+average(e_cpu_infer)+average(e_gpu_infer)+average(e_postprocess));

        if (visible_exp) {
            printf("e_pre : %0.02f, e_infer_cpu : %0.02f, e_infer_gpu : %0.02f, CPU/N: %0.02f\n", average(e_preprocess), average(e_cpu_infer), average(e_gpu_infer), execution_time_wo_waiting/(MAXCORES - 2));
        }

        R = MAX(average(e_gpu_infer), execution_time_wo_waiting/(MAXCORES - 2));
        optimal_core = (int)ceil(execution_time_wo_waiting / R);
        if (optimal_core > (MAXCORES - 2)) optimal_core = MAXCORES - 2;
        max_execution_time = R * optimal_core;

        if (visible_exp) printf("\n::EXP-2:: GPU-Accel with %d threads with %d gpu-layer\n", optimal_core, gLayer);
        reset_check_jitter();
        pthread_barrier_init(&barrier, NULL, optimal_core);
        for (i = 0; i < optimal_core; i++) {
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
            data[i].num_thread = optimal_core;
            data[i].isTest = false;
            data[i].isSet = true;
            data[i].isReclaiming = false;
            rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
            if (rc) {
                printf("Error: Unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        for (i = 0; i < optimal_core; i++) {
            pthread_join(threads[i], NULL);
        }

	execution_time_wo_waiting = (average(e_preprocess)+average(e_cpu_infer)+average(e_gpu_infer)+average(e_postprocess));

        if (visible_exp) {
            printf("e_pre : %0.02f, e_infer_cpu : %0.02f, e_infer_gpu : %0.02f, CPU/N: %0.02f\n", average(e_preprocess), average(e_cpu_infer), average(e_gpu_infer), execution_time_wo_waiting/(MAXCORES - 2));
        }

        R = MAX(average(e_gpu_infer), execution_time_wo_waiting/(MAXCORES - 2));
        optimal_core = (int)ceil(execution_time_wo_waiting / R);
        if (optimal_core > (MAXCORES - 2)) optimal_core = MAXCORES - 2;
        max_execution_time = R * optimal_core;

        if (visible_exp) printf("\n::EXP-3:: GPU-Accel with %d threads with %d gpu-layer\n", optimal_core, gLayer);
        reset_check_jitter();
        pthread_barrier_init(&barrier, NULL, optimal_core);
        for (i = 0; i < optimal_core; i++) {
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
            data[i].num_thread = optimal_core;
            data[i].isTest = false;
            data[i].isSet = true;
            data[i].isReclaiming = false;
            rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
            if (rc) {
                printf("Error: Unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        for (i = 0; i < optimal_core; i++) {
            pthread_join(threads[i], NULL);
        }
        
        char file_path[256] = "measure/";

        char* model_name = malloc(strlen(cfgfile) + 1);
        strncpy(model_name, cfgfile + 6, (strlen(cfgfile)-10));
        model_name[strlen(cfgfile)-10] = '\0';
        
        strcat(file_path, "gpu-accel-CG/");

        strcat(file_path, model_name);
        strcat(file_path, "/");

        strcat(file_path, "gpu-accel_");

        char gpu_portion[20];
        sprintf(gpu_portion, "%03dglayer", gLayer);
        strcat(file_path, gpu_portion);

        strcat(file_path, ".csv");
        if(write_result_gpu(file_path) == -1) {
            /* return error */
            exit(0);
        }
    }

    if (opt_core > 0) optimal_core = opt_core;

    if (optimal_core < (MAXCORES - 2) && (rLayer > 0)) {
        // =====================RECLAMING=====================
        if (visible_exp) printf("\n::EXP-4:: CPU-Reclaiming with %d threads with %d gpu-layer & %d reclaiming-layer\n", optimal_core, gLayer, rLayer);
        reset_check_jitter();
        pthread_barrier_init(&barrier_reclaiming, NULL, optimal_core);
        for (i = 0; i < optimal_core; i++) {
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
            data[i].num_thread = optimal_core;
            data[i].isTest = true;
            if (opt_core > 0) data[i].isSet = false;
            else data[i].isSet = true;
            data[i].isReclaiming = true;
            rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
            if (rc) {
                printf("Error: Unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        for (i = 0; i < optimal_core; i++) {
            pthread_join(threads[i], NULL);
        }

        // =====================RECLAMING + OPTIMAL_CORE=====================

        execution_time_wo_waiting = (average(e_preprocess)+average(e_cpu_infer)+average(e_gpu_infer)+average(e_reclaim_infer)+average(e_postprocess));

        if (visible_exp) {
        printf("e_pre : %0.02f, e_infer_cpu : %0.02f, e_infer_gpu : %0.02f, e_infer_reclaim : %0.02f, CPU/N: %0.02f\n", average(e_preprocess), average(e_cpu_infer), average(e_gpu_infer), average(e_reclaim_infer), execution_time_wo_waiting/(MAXCORES - 2));
        }

        R = maxOfThree(average(e_gpu_infer), average(e_reclaim_infer), execution_time_wo_waiting/(MAXCORES - 2));
        // R = MAX((average(e_gpu_infer)), (average(e_preprocess)+average(e_cpu_infer)+average(e_gpu_infer)+average(e_postprocess)) / MAXCORES -1); 
        optimal_core = (int)ceil(execution_time_wo_waiting / R);
        if (opt_core > 0 && optimal_core > opt_core) optimal_core = opt_core;
        max_execution_time = R * optimal_core;

        if (visible_exp) printf("\n::EXP-5:: CPU-Reclaiming with %d threads with %d gpu-layer & %d reclaiming-layer\n", optimal_core, gLayer, rLayer);
        pthread_barrier_init(&barrier_reclaiming, NULL, optimal_core);
        reset_check_jitter();
        for (i = 0; i < optimal_core; i++) {
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
            data[i].num_thread = optimal_core;
            data[i].isTest = false;
            data[i].isSet = true;
            data[i].isReclaiming = true;
            rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
            if (rc) {
                printf("Error: Unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        for (i = 0; i < optimal_core; i++) {
            pthread_join(threads[i], NULL);
        }

        execution_time_wo_waiting = (average(e_preprocess)+average(e_cpu_infer)+average(e_gpu_infer)+average(e_reclaim_infer)+average(e_postprocess));

        if (visible_exp) {
        printf("e_pre : %0.02f, e_infer_cpu : %0.02f, e_infer_gpu : %0.02f, e_infer_reclaim : %0.02f, CPU/N: %0.02f\n", average(e_preprocess), average(e_cpu_infer), average(e_gpu_infer), average(e_reclaim_infer), execution_time_wo_waiting/(MAXCORES - 2));
        }

        R = maxOfThree(average(e_gpu_infer), average(e_reclaim_infer), execution_time_wo_waiting/(MAXCORES - 2));
        // R = MAX((average(e_gpu_infer)), (average(e_preprocess)+average(e_cpu_infer)+average(e_gpu_infer)+average(e_postprocess)) / MAXCORES -1); 
        optimal_core = (int)ceil(execution_time_wo_waiting / R);
        if (opt_core > 0 && optimal_core > opt_core) optimal_core = opt_core;
        max_execution_time = R * optimal_core;

        if (visible_exp) printf("\n::EXP-6:: CPU-Reclaiming with %d threads with %d gpu-layer & %d reclaiming-layer\n", optimal_core, gLayer, rLayer);
        pthread_barrier_init(&barrier_reclaiming, NULL, optimal_core);
        reset_check_jitter();
        for (i = 0; i < optimal_core; i++) {
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
            data[i].num_thread = optimal_core;
            data[i].isTest = false;
            data[i].isSet = true;
            data[i].isReclaiming = true;
            rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
            if (rc) {
                printf("Error: Unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        for (i = 0; i < optimal_core; i++) {
            pthread_join(threads[i], NULL);
        }
        
        execution_time_wo_waiting = (average(e_preprocess)+average(e_cpu_infer)+average(e_gpu_infer)+average(e_reclaim_infer)+average(e_postprocess));

        if (visible_exp) {
        printf("e_pre : %0.02f, e_infer_cpu : %0.02f, e_infer_gpu : %0.02f, e_infer_reclaim : %0.02f, CPU/N: %0.02f\n", average(e_preprocess), average(e_cpu_infer), average(e_gpu_infer), average(e_reclaim_infer), execution_time_wo_waiting/(MAXCORES - 2));
        }

        R = maxOfThree(average(e_gpu_infer), average(e_reclaim_infer), execution_time_wo_waiting/(MAXCORES - 2));
        // R = MAX((average(e_gpu_infer)), (average(e_preprocess)+average(e_cpu_infer)+average(e_gpu_infer)+average(e_postprocess)) / MAXCORES -1); 
        optimal_core = (int)ceil(execution_time_wo_waiting / R);
        if (opt_core > 0 && optimal_core > opt_core) optimal_core = opt_core;
        max_execution_time = R * optimal_core;

        if (visible_exp) printf("\n::EXP-7:: CPU-Reclaiming with %d threads with %d gpu-layer & %d reclaiming-layer\n", optimal_core, gLayer, rLayer);
        pthread_barrier_init(&barrier_reclaiming, NULL, optimal_core);
        reset_check_jitter();
        for (i = 0; i < optimal_core; i++) {
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
            data[i].num_thread = optimal_core;
            data[i].isTest = false;
            data[i].isSet = true;
            data[i].isReclaiming = true;
            rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
            if (rc) {
                printf("Error: Unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        for (i = 0; i < optimal_core; i++) {
            pthread_join(threads[i], NULL);
        }
        char file_path_[256] = "measure/";

        char* model_name_ = malloc(strlen(cfgfile) + 1);
        strncpy(model_name_, cfgfile + 6, (strlen(cfgfile)-10));
        model_name_[strlen(cfgfile)-10] = '\0';
        

        strcat(file_path_, "cpu-reclaiming-CRG/");
        strcat(file_path_, model_name_);
        strcat(file_path_, "/");

        char gpu_portion_[20];
        sprintf(gpu_portion_, "%dglayer/", gLayer);
        strcat(file_path_, gpu_portion_);

        strcat(file_path_, "cpu-reclaiming_");

        char reclaim_portion[20];
        sprintf(reclaim_portion, "%03drlayer", rLayer);
        strcat(file_path_, reclaim_portion);

        strcat(file_path_, ".csv");
        if(write_result_reclaiming(file_path_) == -1) {
            /* return error */
            exit(0);
        }
    }

    pthread_barrier_destroy(&barrier);
    pthread_barrier_destroy(&barrier_reclaiming);
    
    // pthread_mutex_destroy(&mutex);
    // pthread_cond_destroy(&cond);

    // }
    // else
    // {
    //     printf("\n\ne_infer : %0.02f,e_infer_gpu : %0.02f, e_infer_reclaim : %0.02f, e_infer_cpu : %0.02f\n", average(e_infer), average(e_gpu_infer), average(e_reclaim_infer), average(e_cpu_infer));
    //     printf("\nError: Reclaiming inference time exceeds GPU inference time. Please check your configurations.\n\n");
    // }
    return 0;

}
#else

void cpu_reclaiming_CRG(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    printf("!!ERROR!! GPU = 0 \n");
}
#endif  // GPU
