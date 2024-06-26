#include <stdlib.h>
#include "darknet.h"
#include "network.h"
#include "parser.h"
#include "detector.h"
#include "option_list.h"

#define _GNU_SOURCE
#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <errno.h>  

#ifdef OPENBLAS
#include <cblas.h>
#endif

#ifdef NVTX
#include "nvToolsExt.h"
#endif

#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#ifdef MULTI_PROCESSOR

#define STARTIDX 0
#define START_SYNC 5

static int sem_id;
static key_t key = 1234;

int *start_counter;
int *gpu_counter;
int *reclaim_counter;

typedef struct process_data_t{
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
    int process_id;
    double R;
    double max_preprocess;
    double max_gpu_infer;
    double max_reclaim_infer;
    double max_execution;
    int num_process;
    bool isTest;

} process_data_t;

typedef struct measure_data_t{
    double start_preprocess[200];
    double end_preprocess[200];
    double e_preprocess[200];


    double start_infer[200];
    double start_gpu_waiting[200];
    double start_gpu_infer[200];
    double end_gpu_infer[200];
    double start_reclaim_waiting[200];
    double start_reclaim_infer[1000];
    double end_reclaim_infer[1000];
    double start_cpu_infer[200];
    double end_cpu_infer[200];
    double end_infer[200];

    double waiting_gpu[200];
    double e_gpu_infer[200];
    double waiting_reclaim[1000];
    double e_reclaim_infer[1000];
    double e_cpu_infer[200];
    double e_infer[200];

    double start_postprocess[200];
    double end_postprocess[200];
    double e_postprocess[200];
    double execution_time[200];

    double e_preprocess_max[200];
    double e_gpu_infer_max[200];
    double e_reclaim_infer_max[200];
    double execution_time_max[200];

    double e_preprocess_max_value[200];
    double e_gpu_infer_max_value[200];
    double e_reclaim_infer_max_value[200];
    double execution_time_max_value[200];

    double frame_rate[200];
    double cycle_time[200];
    double start_gap[200];
} measure_data_t;

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

ssize_t write_full(int fd, const void *buffer, size_t count) {
    const char *ptr = (const char *)buffer;
    size_t total_written = 0;
    ssize_t bytes_written;

    while (total_written < count) {
        bytes_written = write(fd, ptr + total_written, count - total_written);
        if (bytes_written == -1) {
            if (errno == EINTR) continue;  // 시그널에 의해 중단된 경우 다시 시도
            perror("write");
            return -1;  // 오류 발생
        }
        total_written += bytes_written;
    }
    return total_written;
}

ssize_t read_full(int fd, void *buffer, size_t count) {
    size_t total_read = 0;
    ssize_t bytes_read;

    while (total_read < count) {
        bytes_read = read(fd, buffer + total_read, count - total_read);
        if (bytes_read == -1) {
            if (errno == EINTR) {  // read가 인터럽트된 경우
                continue;
            }
            perror("read");
            return -1;
        }
        if (bytes_read == 0) {
            break;  // 파일의 끝에 도달했거나 더 이상 읽을 데이터가 없는 경우
        }
        total_read += bytes_read;
    }
    return total_read;
}

static int write_result(char *file_path, measure_data_t *measure_data, int num_exp, int num_process) 
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
    else printf("\nWrite output in %s\n", file_path); 

    double sum_measure_data[num_exp * num_process][33];
    for(i = 0; i < num_exp * num_process; i++)
    {
        int core_id = (i + 1) - (i / num_process) * num_process;
        int count = i / num_process;

        sum_measure_data[i][0] = core_id; // coreIDOrder[core_id];
        sum_measure_data[i][1] = measure_data[core_id - 1].start_preprocess[count];
        sum_measure_data[i][2] = measure_data[core_id - 1].e_preprocess[count];
        sum_measure_data[i][3] = measure_data[core_id - 1].end_preprocess[count];
        sum_measure_data[i][4] = measure_data[core_id - 1].e_preprocess_max[count];
        sum_measure_data[i][5] = measure_data[core_id - 1].e_preprocess_max_value[count];
        sum_measure_data[i][6] = measure_data[core_id - 1].start_infer[count]; 
        sum_measure_data[i][7] = measure_data[core_id - 1].start_gpu_waiting[count];
        sum_measure_data[i][8] = measure_data[core_id - 1].waiting_gpu[count];
        sum_measure_data[i][9] = measure_data[core_id - 1].start_gpu_infer[count];
        sum_measure_data[i][10] = measure_data[core_id - 1].e_gpu_infer[count];
        sum_measure_data[i][11] = measure_data[core_id - 1].end_gpu_infer[count];
        sum_measure_data[i][12] = measure_data[core_id - 1].e_gpu_infer_max[count];
        sum_measure_data[i][13] = measure_data[core_id - 1].e_gpu_infer_max_value[count];
        sum_measure_data[i][14] = measure_data[core_id - 1].waiting_reclaim[count];
        sum_measure_data[i][15] = measure_data[core_id - 1].start_reclaim_infer[count];    
        sum_measure_data[i][16] = measure_data[core_id - 1].e_reclaim_infer[count];    
        sum_measure_data[i][17] = measure_data[core_id - 1].end_reclaim_infer[count];
        sum_measure_data[i][18] = measure_data[core_id - 1].e_reclaim_infer_max[count];
        sum_measure_data[i][19] = measure_data[core_id - 1].e_reclaim_infer_max_value[count];
        sum_measure_data[i][20] = measure_data[core_id - 1].start_cpu_infer[count];
        sum_measure_data[i][21] = measure_data[core_id - 1].e_cpu_infer[count];
        sum_measure_data[i][22] = measure_data[core_id - 1].end_cpu_infer[count];
        sum_measure_data[i][23] = measure_data[core_id - 1].e_infer[count];
        sum_measure_data[i][24] = measure_data[core_id - 1].start_postprocess[count];
        sum_measure_data[i][25] = measure_data[core_id - 1].e_postprocess[count];
        sum_measure_data[i][26] = measure_data[core_id - 1].end_postprocess[count];
        sum_measure_data[i][27] = measure_data[core_id - 1].execution_time[count];
        sum_measure_data[i][28] = measure_data[core_id - 1].execution_time_max[count];
        sum_measure_data[i][29] = measure_data[core_id - 1].execution_time_max_value[count];
        sum_measure_data[i][30] = measure_data[core_id - 1].frame_rate[count];
        sum_measure_data[i][31] = measure_data[core_id - 1].cycle_time[count];
        sum_measure_data[i][32] = measure_data[core_id - 1].start_gap[count]; // start_gap
    }

    qsort(sum_measure_data, sizeof(sum_measure_data)/sizeof(sum_measure_data[0]), sizeof(sum_measure_data[0]), compare);

    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
            "core_id", 
            "start_preprocess", "e_preprocess", "end_preprocess", "e_preprocess_max", "e_preprocess_max_value", 
            "start_infer", 
            "start_gpu_waiting", "waiting_gpu", 
            "start_gpu_infer", "e_gpu_infer", "end_gpu_infer", "e_gpu_infer_max", "e_gpu_infer_max_value", 
            "waiting_reclaim",
            "start_reclaim_infer", "e_reclaim_infer", "end_reclaim_infer", "e_reclaim_infer_max", "e_reclaim_infer_max_value",  
            "start_cpu_infer", "e_cpu_infer", "end_cpu_infer", 
            "e_infer",
            "start_postprocess", "e_postprocess", "end_postprocess", 
            "execution_time", "execution_time_max", "execution_time_max_value", "frame_rate", "cycle_time", "start_gap");

    for(i = 0; i < num_exp * num_process; i++)
    {
        int core_id = (i + 1) - (i / num_process) * num_process;
        double gap = i / num_process;
        if (i == 0) gap = 0.0;
        else gap = sum_measure_data[i][1] - sum_measure_data[i-1][1]; // start_gap

        fprintf(fp, "%0.0f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f\n",  
                sum_measure_data[i][0], sum_measure_data[i][1], sum_measure_data[i][2], sum_measure_data[i][3], 
                sum_measure_data[i][4], sum_measure_data[i][5], sum_measure_data[i][6], sum_measure_data[i][7], 
                sum_measure_data[i][8], sum_measure_data[i][9], sum_measure_data[i][10], sum_measure_data[i][11], 
                sum_measure_data[i][12], sum_measure_data[i][13], sum_measure_data[i][14], sum_measure_data[i][15],
                sum_measure_data[i][16], sum_measure_data[i][17], sum_measure_data[i][18], sum_measure_data[i][19], sum_measure_data[i][20], sum_measure_data[i][21],
                sum_measure_data[i][22], sum_measure_data[i][23], sum_measure_data[i][24], sum_measure_data[i][25],
                sum_measure_data[i][26], sum_measure_data[i][27], sum_measure_data[i][28], sum_measure_data[i][29],
                sum_measure_data[i][30], sum_measure_data[i][31],  gap);
    }

    fclose(fp);

    return 1;
}

static union semun {
    int val;
    struct semid_ds *buf;
    unsigned short *array;
};

static void lock_resource(int resource_num) {
    struct sembuf operations[1];
    operations[0].sem_num = resource_num;  
    operations[0].sem_op = -1;  
    operations[0].sem_flg = 0;  

    if (semop(sem_id, operations, 1) == -1) {
        perror("semop - lock_resource");
        exit(1);
    }
}

static void unlock_resource(int resource_num) {
    struct sembuf operations[1];
    operations[0].sem_num = resource_num;  
    operations[0].sem_op = 1;   
    operations[0].sem_flg = 0;  

    if (semop(sem_id, operations, 1) == -1) {
        perror("semop - unlock_resource");
        exit(1);
    }
}

#ifdef GPU
static void processFunc(process_data_t data, int write_fd)
{
    measure_data_t measure_data;
    // __CPU AFFINITY SETTING__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(data.process_id, &cpuset); // cpu core index
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
    int openblas_thread;

    network net = parse_network_cfg_custom(data.cfgfile, 1, 1, device); // set batch=1
    layer l = net.layers[net.n - 1];

    if (data.weightfile) {
        load_weights(&net, data.weightfile);
    }
    if (net.letter_box) data.letter_box = 1;
    net.benchmark_layers = data.benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    static  int skip_layers[1000][10];
    static int skipped_layers[1000] = {0, };

    for(i = gLayer; i < net.n; i++) {
        for(j = 0; j < 10; j++) {
            if((skip_layers[i][j] < gLayer)&&(skip_layers[i][j] != 0)) {
                skipped_layers[skip_layers[i][j]] = 1;
                // printf("skip layer[%d][%d] : %d,  \n", i, j, skip_layers[i][j]);
            }
        }
    }

    srand(2222222);

    if (data.filename) strncpy(input, data.filename, 256);
    else printf("Error! File is not exist.");

    double remaining_time = 0.0;

    for (i = 0; i < num_exp; i++) {

        // pthread_barrier_wait
        if (i > 0) {
            while(!(*start_counter == data.num_process)) {
                usleep(1);
            }
        }

        // __Preprocess__
        measure_data.start_preprocess[i] = get_time_in_ms();

#ifdef NVTX
        char task[100];
        sprintf(task, "Task (cpu: %d)", data.process_id);
        nvtxRangeId_t nvtx_task;
        nvtx_task = nvtxRangeStartA(task);
#endif

        // printf("\n%d -- Process %d (%d) \n\n", i, data.process_id, sched_getcpu());

        im = load_image(input, 0, 0, net.c);
        resized = resize_min(im, net.w);
        cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        X = cropped.data;

        measure_data.end_preprocess[i] = get_time_in_ms();
        measure_data.e_preprocess[i] = measure_data.end_preprocess[i] - measure_data.start_preprocess[i];

        measure_data.e_preprocess_max_value[i] = data.max_preprocess;
        // Jitter compensation for Preprocessing
        // if (data.isTest) {
        //     remaining_time = data.max_preprocess - (get_time_in_ms() - measure_data.start_preprocess[i]);
        //     if (remaining_time > 0) usleep(remaining_time * 1000);
        // }

        measure_data.e_preprocess_max[i] = get_time_in_ms() - measure_data.start_preprocess[i];

        // __Inference__
        // if (device) predictions = network_predict(net, X);
        // else predictions = network_predict_cpu(net, X);

        measure_data.start_infer[i] = get_time_in_ms();

        if (net.gpu_index != cuda_get_device())
            cuda_set_device(net.gpu_index);
        int size = get_network_input_size(net) * net.batch;
        network_state state;
        state.index = 0;
        state.net = net;
        // state.input = X;
        state.input = net.input_state_gpu;
        memcpy(net.input_pinned_cpu, X, size * sizeof(float));
        state.truth = 0;
        state.train = 0;
        state.delta = 0;

        measure_data.start_gpu_waiting[i] = get_time_in_ms();

        // GPU Inference

        while (!(data.process_id == (*gpu_counter+1))){
            usleep(1);
        }

        lock_resource(0); // 0.2s
        //printf("Process %d is GPU lock\n", data.process_id);
        // printf("%d (%d) -- start_counter = %d, gpu_counter = %d, reclaim_counter = %d\n", i, sched_getcpu(), *start_counter, *gpu_counter, *reclaim_counter);

#ifdef NVTX
        char task_gpu[100];
        sprintf(task_gpu, "Task (cpu: %d) - GPU Inference", data.process_id);
        nvtxRangeId_t nvtx_task_gpu;
        nvtx_task_gpu = nvtxRangeStartA(task_gpu);
#endif

        measure_data.start_gpu_infer[i] = get_time_in_ms();

        cuda_push_array(state.input, net.input_pinned_cpu, size);
        state.workspace = net.workspace;
        for(j = 0; j < gLayer; ++j){
            state.index = j;
            l = net.layers[j];
            if(l.delta_gpu && state.train){
                fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
            }

            l.forward_gpu(l, state);
            if (skipped_layers[j]){
                cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
            }
            state.input = l.output_gpu;
        }

        cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
        state.input = l.output;

        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_gpu);
#endif

        measure_data.end_gpu_infer[i] = get_time_in_ms();

        measure_data.e_gpu_infer_max_value[i] = data.max_gpu_infer;
        // Jitter compensation for GPU inference
        // if (data.isTest) {
        //     // printf("data.max_gpu_infer : %.3f\n", data.max_gpu_infer);
        //     remaining_time = data.max_gpu_infer - (get_time_in_ms() - measure_data.start_gpu_waiting[i]); // [+] Waiting_GPU Time !!
        //     // printf("remaining_time : %.3f\n", remaining_time);
        //     if (remaining_time > 0) usleep(remaining_time * 1000);
        // }
        measure_data.e_gpu_infer_max[i] = get_time_in_ms() - measure_data.start_gpu_infer[i];

        (*gpu_counter)++;

        unlock_resource(0);

        // Reclaiming Inference
        measure_data.start_reclaim_waiting[i] = get_time_in_ms();

#ifdef NVTX
        char task_reclaiming[100];
        sprintf(task_reclaiming, "Task (cpu: %d) - Reclaiming Inference", data.process_id);
        nvtxRangeId_t nvtx_task_reclaiming;
        nvtx_task_reclaiming = nvtxRangeStartA(task_reclaiming);
#endif

        while (!(data.process_id == (*reclaim_counter+1))){
            usleep(1);
        }

        lock_resource(1);
        //printf("Process %d is Reclaim lock\n", data.process_id);
        // printf("%d (%d) -- start_counter = %d, gpu_counter = %d, reclaim_counter = %d\n", i, sched_getcpu(), *start_counter, *gpu_counter, *reclaim_counter);

        measure_data.start_reclaim_infer[i] = get_time_in_ms();

        // Openblas set num threads for Reclaiming inference
        // when cpu reclaim over gpu, exit() okay??????????????????????
        openblas_thread = (MAXCORES-1) - data.num_process + 1;
        openblas_set_num_threads(openblas_thread);
        CPU_ZERO(&cpuset);
        CPU_SET(data.process_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        for (int k = 0; k < openblas_thread - 1; k++) {
            CPU_ZERO(&cpuset);
            CPU_SET(num_process - k, &cpuset);
            openblas_setaffinity(k, sizeof(cpuset), &cpuset);
        }

        state.workspace = net.workspace_cpu;
        gpu_yolo = 0;

        for(j = gLayer; j < rLayer; ++j){
            state.index = j;
            l = net.layers[j];
            if(l.delta && state.train && l.train){
                scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
            }
            l.forward(l, state);
            state.input = l.output;
        }

        measure_data.e_reclaim_infer_max_value[i] = data.max_reclaim_infer;

        // Jitter compensation for Reclaiming inference
        // if (data.isTest) {
        //     remaining_time = data.max_reclaim_infer - (get_time_in_ms() - measure_data.start_reclaim_waiting[i]); // [+] Waiting_GPU Time !!
        //     // remaining_time = data.max_recalim_infer - (get_time_in_ms() - measure_data.start_reclaim_infer[i]); // [+] Waiting_GPU Time !!
        //     if (remaining_time > 0) usleep(remaining_time * 1000);
        // }
        measure_data.e_reclaim_infer_max[i] = get_time_in_ms() - measure_data.start_reclaim_infer[i];
        //printf("Process %d is Reclaiming unlock\n", data.process_id);

        (*reclaim_counter)++;

        unlock_resource(1);

        measure_data.end_reclaim_infer[i] = get_time_in_ms();

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_reclaiming);
#endif

        // CPU Inference
        measure_data.start_cpu_infer[i] = get_time_in_ms();

        openblas_set_num_threads(1);
        CPU_ZERO(&cpuset);
        CPU_SET(data.process_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        for(j = rLayer; j < net.n; ++j){
            //printf("get num threads : %d \n", openblas_get_num_threads());
            state.index = j;
            l = net.layers[j];
            if(l.delta && state.train && l.train){
                scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
            }
            l.forward(l, state);
            state.input = l.output;
        }

        if (gLayer == net.n) predictions = get_network_output_gpu(net);
        else predictions = get_network_output(net, 0);
        reset_wait_stream_events();
        //cuda_free(state.input);   // will be freed in the free_network()

        measure_data.end_cpu_infer[i] = get_time_in_ms();
        measure_data.end_infer[i] = get_time_in_ms();
        measure_data.waiting_gpu[i] = measure_data.start_gpu_infer[i] - measure_data.start_gpu_waiting[i];
        measure_data.e_gpu_infer[i] = measure_data.end_gpu_infer[i] - measure_data.start_gpu_infer[i];
        measure_data.waiting_reclaim[i] = measure_data.start_reclaim_infer[i] - measure_data.start_reclaim_waiting[i];
        measure_data.e_reclaim_infer[i] = measure_data.end_reclaim_infer[i] - measure_data.start_reclaim_infer[i];
        measure_data.e_cpu_infer[i] = measure_data.end_cpu_infer[i] - measure_data.start_cpu_infer[i];
        measure_data.e_infer[i] = measure_data.end_infer[i] - measure_data.start_infer[i];

        // __Postprecess__
        measure_data.start_postprocess[i] = get_time_in_ms();

        // __NMS & TOP acccuracy__
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
                if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
                
                // Print accuracy
                // else printf("%s: %f\n",names[index], predictions[index]);

            }
        }

        // __Display__
        // if (!data.dont_show) {
        //     show_image(im, "predictions");
        //     wait_key_cv(1);
        // }

        // free memory
        free_image(im);
        free_image(resized);
        free_image(cropped);

        // printf("\n%s: Predicted in %0.3f milli-seconds.\n", input, measure_data.e_infer[i]);

        measure_data.end_postprocess[i] = get_time_in_ms();
        measure_data.e_postprocess[i] = measure_data.end_postprocess[i] - measure_data.start_postprocess[i];
        measure_data.execution_time[i] = measure_data.end_postprocess[i] - measure_data.start_preprocess[i];
        measure_data.cycle_time[i] = data.R;
        measure_data.frame_rate[i] = 1000 / data.R;
        measure_data.start_gap[i] = 0;

        measure_data.execution_time_max_value[i] = data.R * data.num_process;
        // Jitter compensation for R
        // if (data.isTest) {
        //     remaining_time = (data.R * data.num_process  - (get_time_in_ms() - measure_data.start_preprocess[i]));
        //     if (remaining_time > 0) usleep(remaining_time * 1000);
        // }

        measure_data.execution_time_max[i] = get_time_in_ms() - measure_data.start_preprocess[i];

        
        lock_resource(2);
        if (*start_counter == data.num_process && *start_counter != 0) *start_counter = 0;
        (*start_counter)++;
        if (*start_counter == data.num_process) {
            *gpu_counter = 0;
            *reclaim_counter = 0;
        }
        unlock_resource(2);


#ifdef NVTX
        nvtxRangeEnd(nvtx_task);
#endif

    }

    // write(write_fd, &measure_data, sizeof(measure_data_t));
    ssize_t nbytes;
    nbytes = write_full(write_fd, &measure_data, sizeof(measure_data_t));
    if (nbytes != sizeof(measure_data_t)) {
        fprintf(stderr, "write error: expected %lu, got %zd\n", sizeof(measure_data_t), nbytes);
    }


    // free memory
    free_detections(dets, nboxes);
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_alphabet(alphabet);
    free_network(net);
    // free_network(net); // Error occur
}

void cpu_reclaiming_mp(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    // Pre-test
    int i, j;
    if (num_process > 11) num_process = 11;

    pid_t pids[num_process];
    int status;

    key_t key1, key2, key3;
    int shm_id1, shm_id2, shm_id3;
    
    // system("touch shmfile"); // Create shmfile
    FILE *fp = fopen("shmfile", "w");
    if (fp == NULL) {
        perror("Failed to create file");
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    key1 = ftok("shmfile", 65);
    key2 = ftok("shmfile", 66);
    key3 = ftok("shmfile", 67);

    printf("\nKey1: %d, Key2: %d, Key3: %d\n", key1, key2, key3); // 키 값 로깅

    int shm_size = 4096;

    // Shared memory 1
    shm_id1 = shmget(key1, shm_size, 0666 | IPC_CREAT);
    if (shm_id1 == -1) {
        perror("shmget failed for key1");
        exit(1);
    }
    start_counter = (int*) shmat(shm_id1, NULL, 0);

    // Shared memory 2
    shm_id2 = shmget(key2, shm_size, 0666 | IPC_CREAT);
    if (shm_id2 == -1) {
        perror("shmget failed for key2");
        exit(1);
    }
    gpu_counter = (int*) shmat(shm_id2, NULL, 0);

    // Shared memory 3
    shm_id3 = shmget(key3, shm_size, 0666 | IPC_CREAT);
    if (shm_id3 == -1) {
        perror("shmget failed for key3");
        exit(1);
    }
    reclaim_counter = (int*) shmat(shm_id3, NULL, 0);

    // Initialize shared memory
    *start_counter = 0;
    *gpu_counter = 0;
    *reclaim_counter = 0;

    // Check value
    (*gpu_counter)++;
    printf("Start Counter: %d\n", *start_counter);
    printf("GPU Counter: %d\n", *gpu_counter);
    printf("Reclaim Counter: %d\n", *reclaim_counter);
    
    // Create 4 semaphore set for lock_resource
    sem_id = semget(key, 4, IPC_CREAT | 0666);

    if (sem_id == -1) {
        perror("semget");
        exit(1);
    }

    // Initialize 4 semaphores
    union semun arg;
    unsigned short values[4] = {1, 1, 1, 1};
    arg.array = values;
    if (semctl(sem_id, 0, SETALL, arg) == -1) {
        perror("semctl - SETALL failed");
        exit(1);
    }

    // Sync :: optimal_core = 11 process
    int optimal_core = 6;
    *start_counter = 0;
    *gpu_counter = 0;
    *reclaim_counter = 0;


    printf("\n::Sync:: CPU-Reclaiming-MP with %d processes with %d gpu-layer & %d reclaim-layer\n", optimal_core, gLayer, rLayer);
    int fd[optimal_core][2];
    process_data_t data[optimal_core];

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
        data[i].process_id = i + 1;
        data[i].max_preprocess = 0;
        data[i].max_gpu_infer = 0;
        data[i].max_reclaim_infer = 0;
        data[i].max_execution = 0;
        data[i].num_process = optimal_core;
        data[i].isTest = false;
    }

    for (i = 0; i < optimal_core; i++) {

        if (pipe(fd[i]) == -1) {
            perror("pipe");
            exit(1);
        }

        pids[i] = fork();
        if (pids[i] == 0) { // child process

            close(fd[i][0]); // close reading end in the child
            processFunc(data[i], fd[i][1]);
            close(fd[i][1]);

            exit(0);
        } else if (pids[i] < 0) {
            perror("fork");
            exit(1);
        }
    }

    measure_data_t receivedData[optimal_core];

    // In the parent process, read data from all child processes
    for (i = 0; i < optimal_core; i++) {
        close(fd[i][1]); // close writing end in the parent
        // read(fd[i][0], &receivedData[i], sizeof(measure_data_t));
        if (read_full(fd[i][0], &receivedData[i], sizeof(measure_data_t)) != sizeof(measure_data_t)) {
            fprintf(stderr, "Failed to read the expected amount of data\n");
        }
        close(fd[i][0]);
    }

    for (i = 0; i < optimal_core; i++) {
        wait(&status);
    }

    char file_path[256] = "measure/";

    char* model_name = malloc(strlen(cfgfile) + 1);
    strncpy(model_name, cfgfile + 6, (strlen(cfgfile)-10));
    model_name[strlen(cfgfile)-10] = '\0';
    
    strcat(file_path, "cpu-reclaiming-mp/");
    strcat(file_path, model_name);
    strcat(file_path, "/");

    char gpu_portion[20];
    sprintf(gpu_portion, "%dglayer/", gLayer);
    strcat(file_path, gpu_portion);

    strcat(file_path, "cpu-reclaiming-mp_");

    char reclaim_portion[20];
    sprintf(reclaim_portion, "%03drlayer", rLayer);
    strcat(file_path, reclaim_portion);

    strcat(file_path, ".csv");

    if(write_result(file_path, receivedData, num_exp, optimal_core) == -1) {
        /* return error */
        exit(0);
    }

    // Detach shared memory segments
    if (shmdt(start_counter) == -1) {
        perror("shmdt start_counter");
    }
    if (shmdt(gpu_counter) == -1) {
        perror("shmdt gpu_counter");
    }
    if (shmdt(reclaim_counter) == -1) {
        perror("shmdt reclaim_counter");
    }

    // Remove shared memory segments
    if (shmctl(shm_id1, IPC_RMID, NULL) == -1) {
        perror("shmctl IPC_RMID shm_id1");
    }
    if (shmctl(shm_id2, IPC_RMID, NULL) == -1) {
        perror("shmctl IPC_RMID shm_id2");
    }
    if (shmctl(shm_id3, IPC_RMID, NULL) == -1) {
        perror("shmctl IPC_RMID shm_id3");
    }

    // Remove semaphore set
    if (semctl(sem_id, 0, IPC_RMID) == -1) {
        perror("semctl IPC_RMID sem_id");
    }


    return 0;

}
#else

void cpu_reclaiming_mp(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    printf("!!ERROR!! GPU = 0 \n");
}
#endif  // GPU
#else

void cpu_reclaiming_mp(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    printf("!!ERROR!! MULTI_PROCESSOR = 0 \n");
}
#endif  // MULTI-PROCESSOR