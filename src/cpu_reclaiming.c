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

static int coreIDOrder[MAXCORES] = {3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};

static pthread_mutex_t mutex_gpu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t mutex_reclaim = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static int current_thread = 1;

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
} thread_data_t;

#ifdef MEASURE
static double core_id_list[1000];
static double start_preprocess[1000];
static double end_preprocess[1000];
static double e_preprocess[1000];

static double start_infer[1000];
static double start_gpu_waiting[1000];
static double start_gpu_infer[1000];
static double end_gpu_infer[1000];
static double start_reclaim_infer[1000];
static double end_reclaim_infer[1000];
static double start_cpu_infer[1000];
static double end_infer[1000];

static double waiting_gpu[1000];
static double e_gpu_infer[1000];
static double waiting_reclaim[1000];
static double e_reclaim_infer[1000];
static double e_cpu_infer[1000];
static double e_infer[1000];


static double start_postprocess[1000];
static double end_postprocess[1000];
static double e_postprocess[1000];

static int optimal_core;
#endif

static double execution_time[1000];
static double frame_rate[1000];

static int is_GPU_larger(double a, double b) {
    return (a - b) >= 2 ? 1 : 0; // Check 2ms differnce
}

static double average(double arr[]){
    double sum;
    int i;
    for(i = 3; i < num_exp; i++) {
        sum += arr[i];
    }
    return sum / (num_exp-3);
}

#ifdef MEASURE

static int compare(const void *a, const void *b) {
    double valueA = *((double *)a + 1);
    double valueB = *((double *)b + 1);

    if (valueA < valueB) return -1;
    if (valueA > valueB) return 1;
    return 0;
}

static int write_result(char *file_path) 
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

    double sum_measure_data[num_exp * optimal_core][24];
    for(i = 0; i < num_exp * optimal_core; i++)
    {
        sum_measure_data[i][0] = core_id_list[i];
        sum_measure_data[i][1] = start_preprocess[i];     
        sum_measure_data[i][2] = e_preprocess[i];       
        sum_measure_data[i][3] = end_preprocess[i];
        sum_measure_data[i][4] = start_infer[i];
        sum_measure_data[i][5] = start_gpu_waiting[i];    
        sum_measure_data[i][6] = waiting_gpu[i];
        sum_measure_data[i][7] = start_gpu_infer[i];       
        sum_measure_data[i][8] = e_gpu_infer[i];        
        sum_measure_data[i][9] = end_gpu_infer[i];
        sum_measure_data[i][10] = waiting_reclaim[i];
        sum_measure_data[i][11] = start_reclaim_infer[i];    
        sum_measure_data[i][12] = e_reclaim_infer[i];    
        sum_measure_data[i][13] = end_reclaim_infer[i];
        sum_measure_data[i][14] = start_cpu_infer[i];     
        sum_measure_data[i][15] = e_cpu_infer[i];     
        sum_measure_data[i][16] = end_infer[i];
        sum_measure_data[i][17] = e_infer[i];
        sum_measure_data[i][18] = start_postprocess[i];     
        sum_measure_data[i][19] = e_postprocess[i];      
        sum_measure_data[i][20] = end_postprocess[i];
        sum_measure_data[i][21] = execution_time[i];          
        sum_measure_data[i][22] = 0.0;      
        sum_measure_data[i][23] = 0.0;
    }

    qsort(sum_measure_data, sizeof(sum_measure_data)/sizeof(sum_measure_data[0]), sizeof(sum_measure_data[0]), compare);

    int startIdx = 10 * optimal_core; // Delete some ROWs
    double new_sum_measure_data[sizeof(sum_measure_data)/sizeof(sum_measure_data[0])-startIdx][sizeof(sum_measure_data[0])];

    int newIndex = 0;
    for (int i = startIdx; i < sizeof(sum_measure_data)/sizeof(sum_measure_data[0]); i++) {
        for (int j = 0; j < sizeof(sum_measure_data[0]); j++) {
            new_sum_measure_data[newIndex][j] = sum_measure_data[i][j];
        }
        newIndex++;
    }

    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
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
            "execution_time", "frame_rate", "optimal_core");

    double frame_rate = 1000 / ( (new_sum_measure_data[(sizeof(new_sum_measure_data)/sizeof(new_sum_measure_data[0]))-1][20]-new_sum_measure_data[0][1]) / (sizeof(new_sum_measure_data)/sizeof(new_sum_measure_data[0])) );

    for(i = 0; i < num_exp * num_thread; i++)
    {
        new_sum_measure_data[i][22] = frame_rate;
        new_sum_measure_data[i][23] = (double)optimal_core;

        fprintf(fp, "%0.0f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.0f\n",  
                new_sum_measure_data[i][0], new_sum_measure_data[i][1], new_sum_measure_data[i][2], 
                new_sum_measure_data[i][3], new_sum_measure_data[i][4], new_sum_measure_data[i][5], 
                new_sum_measure_data[i][6], new_sum_measure_data[i][7], new_sum_measure_data[i][8], 
                new_sum_measure_data[i][9], new_sum_measure_data[i][10], new_sum_measure_data[i][11],
                new_sum_measure_data[i][12], new_sum_measure_data[i][13], new_sum_measure_data[i][14], 
                new_sum_measure_data[i][15], new_sum_measure_data[i][16], new_sum_measure_data[i][17], 
                new_sum_measure_data[i][18], new_sum_measure_data[i][19], new_sum_measure_data[i][20], 
                new_sum_measure_data[i][21], new_sum_measure_data[i][22], new_sum_measure_data[i][23]);
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
    CPU_SET(coreIDOrder[data.thread_id-1], &cpuset); // cpu core index
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
    extern int skip_layers[1000][10];
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

#ifdef MEASURE
        int count = i * data.num_thread + data.thread_id - 1;
#endif

#ifdef NVTX
        char task[100];
        sprintf(task, "Task (cpu: %d)", data.thread_id);
        nvtxRangeId_t nvtx_task;
        nvtx_task = nvtxRangeStartA(task);
#endif

#ifdef MEASURE
        printf("\nThread %d is set to CPU core %d count(%d) : %d \n\n", data.thread_id, sched_getcpu(), data.thread_id, count);
#else
        printf("\nThread %d is set to CPU core %d\n\n", data.thread_id, sched_getcpu());
#endif

        pthread_mutex_lock(&mutex_gpu);

        while(data.thread_id != current_thread) {
            pthread_cond_wait(&cond, &mutex_gpu);
        }

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
        
        // __Inference__
        // if (device) predictions = network_predict(net, X);
        // else predictions = network_predict_cpu(net, X);

#ifdef MEASURE
        start_infer[count] = get_time_in_ms();
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
        start_gpu_waiting[count] = get_time_in_ms();
#endif

        // GPU Inference

#ifdef NVTX
        char task_gpu[100];
        sprintf(task_gpu, "Task (cpu: %d) - GPU Inference", data.thread_id);
        nvtxRangeId_t nvtx_task_gpu;
        nvtx_task_gpu = nvtxRangeStartA(task_gpu);
#endif

#ifdef MEASURE
        start_gpu_infer[count] = get_time_in_ms();
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

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_gpu);
#endif

#ifdef MEASURE
        end_gpu_infer[count] = get_time_in_ms();
#endif

        if (data.thread_id == data.num_thread) {
            current_thread = 1;
        } else {
            current_thread++;
        }

        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mutex_gpu);

        // Reclaiming Inference

        pthread_mutex_lock(&mutex_reclaim);

#ifdef NVTX
        char task_reclaiming[100];
        sprintf(task_reclaiming, "Task (cpu: %d) - Reclaiming Inference", data.thread_id);
        nvtxRangeId_t nvtx_task_reclaiming;
        nvtx_task_reclaiming = nvtxRangeStartA(task_reclaiming);
#endif

#ifdef MEASURE
        start_reclaim_infer[count] = get_time_in_ms();
#endif
        openblas_set_num_threads(3);
        CPU_ZERO(&cpuset);
        CPU_SET(data.thread_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

        CPU_ZERO(&cpuset);
        CPU_SET(10, &cpuset);
        openblas_setaffinity(0, sizeof(cpuset), &cpuset);
        
        CPU_ZERO(&cpuset);
        CPU_SET(11, &cpuset);
        openblas_setaffinity(1, sizeof(cpuset), &cpuset);

        // if(data.num_thread == 1) {
        //     openblas_set_num_threads(1);
        // }
        // else {
        //     openblas_set_num_threads(11 - optimal_core + 1);

        //     CPU_ZERO(&cpuset);
        //     CPU_SET(coreIDOrder[data.thread_id], &cpuset);
        //     pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

        //     for(int i = optimal_core; i < 11; i++){
        //         CPU_ZERO(&cpuset);
        //         CPU_SET(coreIDOrder[i], &cpuset);
        //         openblas_setaffinity(i - optimal_core, sizeof(cpuset), &cpuset);
        //     }

        // }
        
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

        pthread_mutex_unlock(&mutex_reclaim);

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_reclaiming);
#endif

#ifdef MEASURE
        end_reclaim_infer[count] = get_time_in_ms();
#endif

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

        openblas_set_num_threads(1);
        CPU_ZERO(&cpuset);
        CPU_SET(data.thread_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

        for(j = rLayer; j < net.n; ++j){
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

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_cpu);
#endif

#ifdef MEASURE
        end_infer[count] = get_time_in_ms();
        waiting_gpu[count] = start_gpu_infer[count] - start_gpu_waiting[count];
        e_gpu_infer[count] = end_gpu_infer[count] - start_gpu_infer[count];
        waiting_reclaim[count] = start_reclaim_infer[i] - end_gpu_infer[i];
        e_reclaim_infer[count] = end_reclaim_infer[count] - start_reclaim_infer[count];
        e_cpu_infer[count] = end_infer[count] - start_cpu_infer[count];
        e_infer[count] = end_infer[count] - start_infer[count];
        printf("gpu : %0.2f, reclaim : %0.2f, cpu : %0.2f \n", e_gpu_infer[count], e_reclaim_infer[count], e_cpu_infer[count]);
#endif

        // __Postprecess__
#ifdef MEASURE
        start_postprocess[count] = get_time_in_ms();
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
        end_postprocess[count] = get_time_in_ms();
        e_postprocess[count] = end_postprocess[count] - start_postprocess[count];
        execution_time[count] = end_postprocess[count] - start_preprocess[count];
        core_id_list[count] = (double)sched_getcpu();
        // printf("\n%s: Predicted in %0.3f milli-seconds.\n", input, e_infer[count]);
#else
        execution_time[i] = get_time_in_ms() - time;
        frame_rate[i] = 1000.0 / (execution_time[i] / data.num_thread); // N thread
        printf("\n%s: Predicted in %0.3f milli-seconds. (%0.3lf fps)\n", input, execution_time[i], frame_rate[i]);
#endif
        // free memory
        free_image(im);
        free_image(resized);
        free_image(cropped);

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


void cpu_reclaiming(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    
    printf("\n\nCPU-Reclaiming with %d threads with %d gpu-layer & %d reclaim-layer\n", num_thread, gLayer, rLayer);

    pthread_t threads[MAXCORES - 1];
    int rc;
    int i;

    thread_data_t data[MAXCORES - 1];

// #ifdef MEASURE
//     printf("\n\nFinding Optimal Core when CPU-Reclaiming with 1 thread with %d gpu-layer\n", gLayer);

//     for (i = 0; i < 1; i++) {
//         data[i].datacfg = datacfg;
//         data[i].cfgfile = cfgfile;
//         data[i].weightfile = weightfile;
//         data[i].filename = filename;
//         data[i].thresh = thresh;
//         data[i].hier_thresh = hier_thresh;
//         data[i].dont_show = dont_show;
//         data[i].ext_output = ext_output;
//         data[i].save_labels = save_labels;
//         data[i].outfile = outfile;
//         data[i].letter_box = letter_box;
//         data[i].benchmark_layers = benchmark_layers;
//         data[i].thread_id = i + 1;
//         data[i].num_thread = 1;
//         rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
//         if (rc) {
//             printf("Error: Unable to create thread, %d\n", rc);
//             exit(-1);
//         }
//     }

//     for (i = 0; i < 1; i++) {
//         pthread_join(threads[i], NULL);
//         pthread_detach(threads[i]);
//     }

//     if ( is_GPU_larger(average(e_gpu_infer),average(e_reclaim_infer)) )
//     {
//         optimal_core = (int)ceil(average(e_infer) / MAX((average(e_gpu_infer)+average(e_preprocess)),average(e_reclaim_infer)));
//         if(optimal_core > 11) optimal_core = 11;

//         printf("e_pre+e_infer : %0.02f, e_pre+e_infer_gpu : %0.02f, e_infer_reclaim : %0.02f, e_infer_cpu : %0.02f, Optimal Core : %d, CPU/N: %0.02f \n", average(e_infer)+average(e_preprocess), average(e_gpu_infer)+average(e_preprocess), average(e_reclaim_infer), average(e_cpu_infer), optimal_core, average(e_cpu_infer)/optimal_core);

//         printf("\n\nCPU-Reclaiming with %d threads with %d gpu-layer with %d reclaim-layer\n", optimal_core, gLayer, rLayer);
        
//         for (i = 0; i < optimal_core; i++) {
//             data[i].datacfg = datacfg;
//             data[i].cfgfile = cfgfile;
//             data[i].weightfile = weightfile;
//             data[i].filename = filename;
//             data[i].thresh = thresh;
//             data[i].hier_thresh = hier_thresh;
//             data[i].dont_show = dont_show;
//             data[i].ext_output = ext_output;
//             data[i].save_labels = save_labels;
//             data[i].outfile = outfile;
//             data[i].letter_box = letter_box;
//             data[i].benchmark_layers = benchmark_layers;
//             data[i].thread_id = i + 1;
//             data[i].num_thread = optimal_core;
//             rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
//             if (rc) {
//                 printf("Error: Unable to create thread, %d\n", rc);
//                 exit(-1);
//             }
//         }

//         for (i = 0; i < optimal_core; i++) {
//             pthread_join(threads[i], NULL);
//         }
// #else
    for (i = 0; i < num_thread; i++) {
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
        data[i].num_thread = num_thread;
        rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
        if (rc) {
            printf("Error: Unable to create thread, %d\n", rc);
            exit(-1);
        }
    }

    for (i = 0; i < num_thread; i++) {
        pthread_join(threads[i], NULL);
    }
// #endif

#ifdef MEASURE
        char file_path[256] = "measure/";

        char* model_name = malloc(strlen(cfgfile) + 1);
        strncpy(model_name, cfgfile + 6, (strlen(cfgfile)-10));
        model_name[strlen(cfgfile)-10] = '\0';
        

        strcat(file_path, "cpu-reclaiming/");
        strcat(file_path, model_name);
        strcat(file_path, "/");

        char gpu_portion[20];
        sprintf(gpu_portion, "%dglayer/", gLayer);
        strcat(file_path, gpu_portion);

        strcat(file_path, "cpu-reclaiming_");

        char reclaim_portion[20];
        sprintf(reclaim_portion, "%03drlayer", rLayer);
        strcat(file_path, reclaim_portion);

        strcat(file_path, ".csv");
        if(write_result(file_path) == -1) {
            /* return error */
            exit(0);
        }
#endif

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

void cpu_reclaiming(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    printf("!!ERROR!! GPU = 0 \n");
}
#endif  // GPU