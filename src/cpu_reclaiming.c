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

static pthread_mutex_t mutex_gpu = PTHREAD_MUTEX_INITIALIZER;
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
} thread_data_t;

#ifdef MEASURE
static double start_preprocess[1000];
static double end_preprocess[1000];
static double e_preprocess[1000];

static double start_infer[1000];
static double end_infer[1000];
static double e_infer[1000];

static double start_postprocess[1000];
static double end_postprocess[1000];
static double e_postprocess[1000];
#endif

static double execution_time[1000];
static double frame_rate[1000];

#ifdef MEASURE
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

    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
            "core_id", "start_preprocess", "e_preprocess", "end_preprocess", 
            "start_infer", "e_infer", "end_infer", 
            "start_postprocess", "e_postprocess", "end_postprocess", 
            "execution_time", "frame_rate");

    for(i = 0; i < num_exp * num_thread; i++)
    {
        fprintf(fp, "%d,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f\n",  
                (i + 1) - (i / num_thread) * num_thread, 
                start_preprocess[i], e_preprocess[i], end_preprocess[i], 
                start_infer[i], e_infer[i], end_infer[i], 
                start_postprocess[i], e_postprocess[i], end_postprocess[i], 
                execution_time[i], frame_rate[i]);
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
    CPU_SET(data.thread_id, &cpuset); // cpu core index
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
    extern int skip_layers[1000];
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
        int count = i * num_thread + data.thread_id - 1;
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

        // GPU Inference
        pthread_mutex_lock(&mutex_gpu);

        // while(data.thread_id != current_thread) {
        //     pthread_cond_wait(&cond, &mutex_gpu);
        // }

#ifdef NVTX
        char task_gpu[100];
        sprintf(task_gpu, "Task (cpu: %d) - GPU Inference", data.thread_id);
        nvtxRangeId_t nvtx_task_gpu;
        nvtx_task_gpu = nvtxRangeStartA(task_gpu);
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

        if (data.thread_id == num_thread) {
            current_thread = 1;
        } else {
            current_thread++;
        }

        // pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mutex_gpu);

        // Reclaiming Inference
#ifdef NVTX
        char task_reclaiming[100];
        sprintf(task_reclaiming, "Task (cpu: %d) - Reclaiming Inference", data.thread_id);
        nvtxRangeId_t nvtx_task_reclaiming;
        nvtx_task_reclaiming = nvtxRangeStartA(task_reclaiming);
#endif

        openblas_set_num_threads(3);
        CPU_ZERO(&cpuset);
        CPU_SET(data.thread_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

        CPU_ZERO(&cpuset);
        CPU_SET(6, &cpuset);
        openblas_setaffinity(0, sizeof(cpuset), &cpuset);
        
        CPU_ZERO(&cpuset);
        CPU_SET(7, &cpuset);
        openblas_setaffinity(1, sizeof(cpuset), &cpuset);

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

#ifdef NVTX
        nvtxRangeEnd(nvtx_task_reclaiming);
#endif

        // CPU Inference
        openblas_set_num_threads(1);
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

#ifdef MEASURE
        end_infer[count] = get_time_in_ms();
        e_infer[count] = end_infer[count] - start_infer[count];
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
                else printf("%s: %f\n",names[index], predictions[index]);
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
        frame_rate[count] = 1000.0 / execution_time[count];
        printf("\n%s: Predicted in %0.3f milli-seconds.\n", input, e_infer[count]);
#else
        execution_time[i] = get_time_in_ms() - time;
        frame_rate[i] = 1000.0 / (execution_time[i] / num_thread); // N thread
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

    pthread_t threads[num_thread];
    int rc;
    int i;

    thread_data_t data[num_thread];

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
        rc = pthread_create(&threads[i], NULL, threadFunc, &data[i]);
        if (rc) {
            printf("Error: Unable to create thread, %d\n", rc);
            exit(-1);
        }
    }

    for (i = 0; i < num_thread; i++) {
        pthread_join(threads[i], NULL);
    }

#ifdef MEASURE
    printf("!!Write CSV File!! \n");
    char file_path[256] = "measure/";

    char* model_name = malloc(strlen(cfgfile) + 1);
    strncpy(model_name, cfgfile + 6, (strlen(cfgfile)-10));
    model_name[strlen(cfgfile)-10] = '\0';
    

    strcat(file_path, "cpu-reclaiming/");
    strcat(file_path, model_name);
    strcat(file_path, "/");

    strcat(file_path, "cpu-reclaiming");

    strcat(file_path, ".csv");
    if(write_result(file_path) == -1) {
        /* return error */
        exit(0);
    }
#endif

    // pthread_mutex_destroy(&mutex);
    // pthread_cond_destroy(&cond);

    return 0;

}
#else

void cpu_reclaiming(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    printf("!!ERROR!! GPU = 0 \n");
}
#endif  // GPU