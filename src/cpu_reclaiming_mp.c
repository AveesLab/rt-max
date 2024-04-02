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

#ifdef OPENBLAS
#include <cblas.h>
#endif

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

#ifdef MULTI_PROCESSOR

static int sem_id;
static key_t key = 1234;

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

#ifndef MEASURE
    double execution_time[200];
    double frame_rate[200];
#endif

} process_data_t;

#ifdef MEASURE
typedef struct measure_data_t{
    double start_preprocess[200];
    double end_preprocess[200];
    double e_preprocess[200];

    double start_infer[200];
    double start_gpu_waiting[200];
    double start_gpu_infer[200];
    double end_gpu_infer[200];
    double start_reclaim_infer[1000];
    double end_reclaim_infer[1000];
    double start_cpu_infer[200];
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
    double frame_rate[200];

} measure_data_t;
double max_gpu_infer_time = 0.0f;
double max_execution_time = 0.0f;
double avg_gpu_infer_time = 0.0f;
double avg_execution_time = 0.0f;

#endif
int optimal_core = 11;
int num_thread = 0;
double R = 0.0f;
#ifdef MEASURE
static int compare(const void *a, const void *b) {
    double valueA = *((double *)a + 1);
    double valueB = *((double *)b + 1);

    if (valueA < valueB) return -1;
    if (valueA > valueB) return 1;
    return 0;
}

static int write_result(char *file_path, measure_data_t *measure_data) 
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

    double sum_measure_data[num_exp * optimal_core][23];
    for(i = 0; i < num_exp * optimal_core; i++)
    {
        int core_id = (i + 1) - (i / optimal_core) * optimal_core;
        int count = i / optimal_core;
        
        measure_data[core_id - 1].execution_time[count] = measure_data[core_id - 1].end_postprocess[count] - measure_data[core_id - 1].start_preprocess[count];
        measure_data[core_id - 1].frame_rate[count] = 1000 / measure_data[core_id - 1].execution_time[count];

        sum_measure_data[i][0] = (double)core_id;
        sum_measure_data[i][1] = measure_data[core_id - 1].start_preprocess[count];     
        sum_measure_data[i][2] = measure_data[core_id - 1].e_preprocess[count];       
        sum_measure_data[i][3] = measure_data[core_id - 1].end_preprocess[count];
        sum_measure_data[i][4] = measure_data[core_id - 1].start_infer[count];
        sum_measure_data[i][5] = measure_data[core_id - 1].start_gpu_waiting[count];    
        sum_measure_data[i][6] = measure_data[core_id - 1].waiting_gpu[count];
        sum_measure_data[i][7] = measure_data[core_id - 1].start_gpu_infer[count];       
        sum_measure_data[i][8] = measure_data[core_id - 1].e_gpu_infer[count];        
        sum_measure_data[i][9] = measure_data[core_id - 1].end_gpu_infer[count];
        sum_measure_data[i][10] = measure_data[core_id - 1].waiting_reclaim[count];
        sum_measure_data[i][11] = measure_data[core_id - 1].start_reclaim_infer[count];    
        sum_measure_data[i][12] = measure_data[core_id - 1].e_reclaim_infer[count];    
        sum_measure_data[i][13] = measure_data[core_id - 1].end_reclaim_infer[count];
        sum_measure_data[i][14] = measure_data[core_id - 1].start_cpu_infer[count];     
        sum_measure_data[i][15] = measure_data[core_id - 1].e_cpu_infer[count];     
        sum_measure_data[i][16] = measure_data[core_id - 1].end_infer[count];
        sum_measure_data[i][17] = measure_data[core_id - 1].e_infer[count];
        sum_measure_data[i][18] = measure_data[core_id - 1].start_postprocess[count];     
        sum_measure_data[i][19] = measure_data[core_id - 1].e_postprocess[count];      
        sum_measure_data[i][20] = measure_data[core_id - 1].end_postprocess[count];
        sum_measure_data[i][21] = measure_data[core_id - 1].execution_time[count];          
        sum_measure_data[i][22] = measure_data[core_id - 1].frame_rate[count];
    }

    qsort(sum_measure_data, sizeof(sum_measure_data)/sizeof(sum_measure_data[0]), sizeof(sum_measure_data[0]), compare);

    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
            "core_id", 
            "start_preprocess", "e_preprocess", "end_preprocess", 
            "start_infer", 
            "start_gpu_waiting", "waiting_gpu", 
            "start_gpu_infer", "e_gpu_infer", "end_gpu_infer", 
            "waiting_reclaim",
            "start_reclaim_infer", "e_reclaim_infer", "end_reclaim_infer", 
            "start_cpu_infer", "e_cpu_infer", "end_infer", 
            "e_infer",
            "start_postprocess", "e_postprocess", "end_postprocess", 
            "execution_time", "frame_rate");

    for(i = 0; i < num_exp * optimal_core; i++)
    {
        fprintf(fp, "%0.0f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f\n",  
                sum_measure_data[i][0], sum_measure_data[i][1], sum_measure_data[i][2], 
                sum_measure_data[i][3], sum_measure_data[i][4], sum_measure_data[i][5], 
                sum_measure_data[i][6], sum_measure_data[i][7], sum_measure_data[i][8], 
                sum_measure_data[i][9], sum_measure_data[i][10], sum_measure_data[i][11],
                sum_measure_data[i][12], sum_measure_data[i][13], sum_measure_data[i][14], 
                sum_measure_data[i][15], sum_measure_data[i][16], sum_measure_data[i][17], 
                sum_measure_data[i][18], sum_measure_data[i][19], sum_measure_data[i][20], 
                sum_measure_data[i][21], sum_measure_data[i][22]);
    }

    fclose(fp);

    return 1;
}
#endif

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
#ifdef MEASURE
static void processFunc(process_data_t data, int write_fd)
#else
static void processFunc(process_data_t data)
#endif
{
#ifdef MEASURE
    measure_data_t measure_data;
#endif

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

    network net = parse_network_cfg_custom(data.cfgfile, 1, 1, device); // set batch=1
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

    for (i = 0; i < num_exp; i++) {

#ifdef NVTX
        char task[100];
        sprintf(task, "Task (cpu: %d)", data.process_id);
        nvtxRangeId_t nvtx_task;
        nvtx_task = nvtxRangeStartA(task);
#endif

#ifdef MEASURE
        // printf("\nProcess %d is set to CPU core %d count(%d) : %d \n\n", data.process_id, sched_getcpu(), data.process_id, i);
#else
        printf("\nProcess %d is set to CPU core %d\n\n", data.process_id, sched_getcpu());
#endif

        time = get_time_in_ms();
        // __Preprocess__
        lock_resource(0);
#ifdef MEASURE
        measure_data.start_preprocess[i] = get_time_in_ms();
#endif
        im = load_image(input, 0, 0, net.c);
        resized = resize_min(im, net.w);
        cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        X = cropped.data;

#ifdef MEASURE
        measure_data.end_preprocess[i] = get_time_in_ms();
        measure_data.e_preprocess[i] = measure_data.end_preprocess[i] - measure_data.start_preprocess[i];
#endif
        
        // __Inference__
        // if (device) predictions = network_predict(net, X);
        // else predictions = network_predict_cpu(net, X);

#ifdef MEASURE
        measure_data.start_infer[i] = get_time_in_ms();
#endif

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

#ifdef MEASURE
        measure_data.start_gpu_waiting[i] = get_time_in_ms();
#endif

        // GPU Inference

#ifdef NVTX
        char task_gpu[100];
        sprintf(task_gpu, "Task (cpu: %d) - GPU Inference", data.process_id);
        nvtxRangeId_t nvtx_task_gpu;
        nvtx_task_gpu = nvtxRangeStartA(task_gpu);
#endif

#ifdef MEASURE
        measure_data.start_gpu_infer[i] = get_time_in_ms();
#endif

        cuda_push_array(state.input, net.input_pinned_cpu, size);
        state.workspace = net.workspace;
        for(j = 0; j < gLayer; ++j){
            state.index = j;
            l = net.layers[j];
            if(l.delta_gpu && state.train){
                fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
            }

            l.forward_gpu(l, state);
            if (skip_layers[j]){
                cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
            }
            state.input = l.output_gpu;
        }

        cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
        state.input = l.output;

        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

#ifdef MEASURE
        measure_data.end_gpu_infer[i] = get_time_in_ms();
#endif

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_gpu);
#endif

        unlock_resource(0);

        // Reclaiming Inference
#ifdef NVTX
        char task_reclaiming[100];
        sprintf(task_reclaiming, "Task (cpu: %d) - Reclaiming Inference", data.process_id);
        nvtxRangeId_t nvtx_task_reclaiming;
        nvtx_task_reclaiming = nvtxRangeStartA(task_reclaiming);
#endif
        
        lock_resource(1);

#ifdef MEASURE
        measure_data.start_reclaim_infer[i] = get_time_in_ms();
#endif

        // when cpu reclaim over gpu, exit() okay??????????????????????
        num_thread = num_process - optimal_core + 1;
        openblas_set_num_threads(num_thread);
        CPU_ZERO(&cpuset);
        CPU_SET(data.process_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

        for (int k = 0; k < num_thread - 1; k++) {
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

        unlock_resource(1);

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_reclaiming);
#endif

#ifdef MEASURE
        measure_data.end_reclaim_infer[i] = get_time_in_ms();
#endif

        // CPU Inference

#ifdef MEASURE
        measure_data.start_cpu_infer[i] = get_time_in_ms();
#endif
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

#ifdef MEASURE
        measure_data.end_infer[i] = get_time_in_ms();
        measure_data.waiting_gpu[i] = measure_data.start_gpu_infer[i] - measure_data.start_gpu_waiting[i];
        measure_data.e_gpu_infer[i] = measure_data.end_gpu_infer[i] - measure_data.start_gpu_infer[i];
        measure_data.waiting_reclaim[i] = measure_data.start_reclaim_infer[i] - measure_data.end_gpu_infer[i];
        measure_data.e_reclaim_infer[i] = measure_data.end_reclaim_infer[i] - measure_data.start_reclaim_infer[i];
        measure_data.e_cpu_infer[i] = measure_data.end_infer[i] - measure_data.start_cpu_infer[i];
        measure_data.e_infer[i] = measure_data.end_infer[i] - measure_data.start_infer[i];
#endif

        // __Postprecess__
#ifdef MEASURE
        measure_data.start_postprocess[i] = get_time_in_ms();
#endif

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

#ifndef MEASURE
                else printf("%s: %f\n",names[index], predictions[index]);
#endif

            }
        }

        // __Display__
        // if (!data.dont_show) {
        //     show_image(im, "predictions");
        //     wait_key_cv(1);
        // }

#ifdef MEASURE
        measure_data.end_postprocess[i] = get_time_in_ms();
        measure_data.e_postprocess[i] = measure_data.end_postprocess[i] - measure_data.start_postprocess[i];
        measure_data.execution_time[i] = measure_data.end_postprocess[i] - measure_data.start_preprocess[i];
        measure_data.frame_rate[i] = 1000.0 / measure_data.execution_time[i];
        // printf("\n%s: Predicted in %0.3f milli-seconds.\n", input, measure_data.e_infer[i]);
#else
        data.execution_time[i] = get_time_in_ms() - time;
        data.frame_rate[i] = 1000.0 / (data.execution_time[i] / num_process); // N process
        printf("\n%s: Predicted in %0.3f milli-seconds. (%0.3lf fps)\n", input, data.execution_time[i], data.frame_rate[i]);
#endif

        // free memory
        free_image(im);
        free_image(resized);
        free_image(cropped);

#ifdef NVTX
        nvtxRangeEnd(nvtx_task);
#endif
    }

#ifdef MEASURE
    write(write_fd, &measure_data, sizeof(measure_data_t));
#endif

    // free memory
    free_detections(dets, nboxes);
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_alphabet(alphabet);
    // free_network(net);
}


void cpu_reclaiming_mp(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{

    printf("\n\nCPU-Reclaiming-MP with %d processes with %d gpu-layer & %d reclaim-layer\n", num_process, gLayer, rLayer);

    int i, j;

    pid_t pids[MAXCORES-1];
    int status;

#ifdef MEASURE
    int fd[MAXCORES-1][2];
#endif

    // Create semaphore set with NUM_PROCESSES semaphores
    sem_id = semget(key, 2, IPC_CREAT | 0666);

    if (sem_id == -1) {
        perror("semget");
        exit(1);
    }

    // Initialize semaphores
    union semun arg;
    unsigned short values[2] = {1, 1};  // Initialize both semaphores to 1
    arg.array = values;
    semctl(sem_id, 0, SETALL, arg);

    process_data_t data[MAXCORES-1];

    for (i = 0; i < num_process; i++) {
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
    }

    for (i = 0; i < num_process; i++) {

#ifdef MEASURE
        if (pipe(fd[i]) == -1) {
            perror("pipe");
            exit(1);
        }
#endif

        pids[i] = fork();
        if (pids[i] == 0) { // child process

#ifdef MEASURE
            close(fd[i][0]); // close reading end in the child
            processFunc(data[i], fd[i][1]);
            close(fd[i][1]);
#else
            processFunc(data[i]);
#endif

            exit(0);
        } else if (pids[i] < 0) {
            perror("fork");
            exit(1);
        }
    }
    for (i = 0; i < num_process; i++) {
        wait(&status);
    }

#ifdef MEASURE
    measure_data_t receivedData[num_process];

    // In the parent process, read data from all child processes
    for (i = 0; i < num_process; i++) {
        close(fd[i][1]); // close writing end in the parent
        read(fd[i][0], &receivedData[i], sizeof(measure_data_t));
        close(fd[i][0]);
    }
#endif

    double max_gpu_infer_time = 0.0;
    double max_execution_time = 0.0;

    for (i = 10; i < 15; i++) {
        printf("max_gpu_infer_time: %lf\n", MAX(max_gpu_infer_time, receivedData[0].e_gpu_infer[i])); // GPU_infer
    }

    printf("\n\n::Test EXP done::");
    int fd2[MAXCORES-1][2];

    int startIdx = 10 * num_process;
    for (i = 10; i < num_exp; i++) {
        for (j = 0; j < num_process; j++) {
            max_gpu_infer_time = MAX(max_gpu_infer_time, receivedData[j].e_gpu_infer[i]);
            max_execution_time = MAX(max_execution_time, (receivedData[j].e_preprocess[i]+receivedData[j].e_gpu_infer[i]+receivedData[j].e_cpu_infer[i]+receivedData[j].e_postprocess[i]));
            avg_gpu_infer_time += receivedData[j].e_gpu_infer[i];
            avg_execution_time += receivedData[j].e_preprocess[i]+receivedData[j].e_gpu_infer[i]+receivedData[j].e_cpu_infer[i]+receivedData[j].e_postprocess[i];
        }        
    }
    avg_gpu_infer_time /= num_process * num_exp - startIdx + 1;
    avg_execution_time /= num_process * num_exp - startIdx + 1;
    double wcet_ratio = 1.07;
    max_gpu_infer_time = avg_gpu_infer_time * wcet_ratio; // GPU_infer
    max_execution_time = avg_execution_time * wcet_ratio; // CPU_infer + Post
    printf("\n\n avg execution time = %f\n\n", avg_execution_time);
    printf("\n\n max execution time = %f\n\n", max_execution_time);
    printf("\n\n avg gpu infer time = %f\n\n", avg_gpu_infer_time);
    printf("\n\n max gpu infer time = %f\n\n", max_gpu_infer_time);


    R = MAX(max_gpu_infer_time, max_execution_time / num_process);
    optimal_core = ceil(max_execution_time / R);
    printf("\n\nOptimal core = %d\n\n", optimal_core);
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
    }
    printf("\n\n::EXP-1:: CPU-Reclaiming-MP with %d processes with %d gpu-layer & %d reclaim-layer\n", num_process, gLayer, rLayer);
    for (i = 0; i < optimal_core; i++) {
#ifdef MEASURE
        if (pipe(fd2[i]) == -1) {
            perror("pipe");
            exit(1);
        }
#endif
        pids[i] = fork();
        if (pids[i] == 0) { // child process
        #ifdef MEASURE
            close(fd2[i][0]); // close reading end in the child
            processFunc(data[i], fd2[i][1]);
            close(fd2[i][1]);
        #else
            processFunc(data[i]);
            #endif

            exit(0);
        } else if (pids[i] < 0) {
            perror("fork");
            exit(1);
        }
    }
    for (i = 0; i < num_process; i++) {
        wait(&status);
    }

#ifdef MEASURE
    // In the parent process, read data from all child processes
    for (i = 0; i < optimal_core; i++) {
        close(fd2[i][1]); // close writing end in the parent
        read(fd2[i][0], &receivedData[i], sizeof(measure_data_t));
        close(fd2[i][0]);
    }
#endif

    // Remove semaphores
    //semctl(sem_id, 0, IPC_RMID);
    max_gpu_infer_time = 0.0;
    max_execution_time = 0.0;

    for (i = 160; i < 165; i++) {
        printf("max_gpu_infer_time: %lf\n", MAX(max_gpu_infer_time, receivedData[0].e_gpu_infer[i])); // GPU_infer
    }

    printf("\n\n EXP-1 done");
    semctl(sem_id, 0, IPC_RMID);
#ifdef MEASURE
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
    if(write_result(file_path, receivedData) == -1) {
        /* return error */
        exit(0);
    }
#endif

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