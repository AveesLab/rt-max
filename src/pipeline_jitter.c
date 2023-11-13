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

#ifdef MEASURE
static double max_time;
static double min_time;

static double start_preprocess[1000];
static double start_preprocess_array[1000];
static double end_preprocess[1000];
static double end_preprocess_array[1000];
static double e_preprocess[1000];

static double start_infer[1000];
static double start_infer_array[1000];
static double end_infer[1000];
static double e_infer[1000];
static double e_infer_max[1000];
static double end_infer_max[1000];
static double end_infer_array[1000];

static double start_postprocess[1000];
static double start_postprocess_array[1000];
static double end_postprocess[1000];
static double e_postprocess[1000];
static double end_postprocess_array[1000];

static double e_stall[1000];
#endif

double remaining_time = 0.0;
double wait_start = 0.0;
double wait_end = 0.0;
double work_time = 0.0;

static double execution_time[1000];
static double frame_rate[1000];

static int thread_index = 0;
static int preprocess_index = 0;
static int inference_index = 0;
static int postprocess_index = 0;

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;
static int demo_letterbox;

static int nboxes = 0;
static detection *dets = NULL;
static float* prediction;
static float* predictions;
static float *avg;

static list *options;
static char *name_list;
static int names_size;
static char **names; 

static char buff[256];
static char *input;

static image **alphabet;

static float nms;    // 0.4F

static network net;

static int top;
static int nboxes;
static int* indexes;

static char *target_model;
static int object_detection;

static float fps = 0;
static float demo_thresh = 0;
static float demo_hier_thresh = 0;
static int demo_ext_output = 0;
static long long int frame_id = 0;
static int demo_json_port = -1;
static bool demo_skip_frame = false;

static int avg_frames;

static int device = 0; // Choose CPU or GPU

static network net; // set batch=1
static layer l;

static image im, resized, cropped;
static float *X, *predictions;
static detection *dets;

static int barrier_signal = 0;
pthread_barrier_t barrier;

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

    double sum_measure_data[num_exp][14];
    for(i = 0; i < num_exp; i++)
    {
        sum_measure_data[i][0] = start_preprocess_array[i];
        sum_measure_data[i][1] = e_preprocess[i];
        sum_measure_data[i][2] = end_preprocess_array[i];
        sum_measure_data[i][3] = start_infer_array[i];
        sum_measure_data[i][4] = e_infer[i];
        sum_measure_data[i][5] = e_infer_max[i];
        sum_measure_data[i][6] = end_infer_array[i];
        sum_measure_data[i][7] = start_postprocess_array[i];
        sum_measure_data[i][8] = e_postprocess[i];
        sum_measure_data[i][9] = end_postprocess_array[i];
        sum_measure_data[i][10] = e_stall[i];
        sum_measure_data[i][11] = execution_time[i];
        sum_measure_data[i][12] = 0.0;
        sum_measure_data[i][13] = 0.0;
        // printf("e_infer : %0.2f \n",e_infer_max[i]);

    }

    int startIdx = 30; // Delete some ROWs
    double new_sum_measure_data[sizeof(sum_measure_data)/sizeof(sum_measure_data[0])-startIdx][sizeof(sum_measure_data[0])];
    int newIndex = 0;
    for (int i = startIdx; i < sizeof(sum_measure_data)/sizeof(sum_measure_data[0]); i++) {
        for (int j = 0; j < sizeof(sum_measure_data[0]); j++) {
            new_sum_measure_data[newIndex][j] = sum_measure_data[i][j];
        }
        newIndex++;
    }
    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
            "start_preprocess",     "e_preprocess",     "end_preprocess", 
            "start_infer",          "e_infer",          "e_infer_max",           "end_infer", 
            "start_postprocess",    "e_postprocess",    "end_postprocess", 
            "e_stall",              "execution_time",      "cycle_time",       "frame_rate");

    double frame_rate = 0.0;
    double cycle_time = 0.0;

    for(i = 0; i < num_exp - startIdx; i++)
    {
        if (i == 0) cycle_time = NAN;
        else cycle_time = new_sum_measure_data[i][0] - new_sum_measure_data[i-1][0];

        if (i == 0) frame_rate = NAN;
        else frame_rate = 1000/cycle_time;

        new_sum_measure_data[i][12] = cycle_time;
        new_sum_measure_data[i][13] = frame_rate;

        fprintf(fp, "%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f\n",  
                new_sum_measure_data[i][0], new_sum_measure_data[i][1], new_sum_measure_data[i][2], new_sum_measure_data[i][3], 
                new_sum_measure_data[i][4], new_sum_measure_data[i][5], new_sum_measure_data[i][6], new_sum_measure_data[i][7], 
                new_sum_measure_data[i][8], new_sum_measure_data[i][9], new_sum_measure_data[i][10], new_sum_measure_data[i][11], new_sum_measure_data[i][12], new_sum_measure_data[i][13]);
    }
    
    return 1;
}
#endif


static void push_data(int i)
{
    start_preprocess_array[i] = start_preprocess[preprocess_index];
    e_preprocess[i] = end_preprocess[preprocess_index] - start_preprocess[preprocess_index];
    end_preprocess_array[i] = end_preprocess[preprocess_index];


    start_infer_array[i] = start_infer[inference_index];
    e_infer[i] = end_infer[inference_index] - start_infer[inference_index];
    e_infer_max[i] = end_infer_max[inference_index] - start_infer[inference_index];
    end_infer_array[i] = end_infer[inference_index];

    start_postprocess_array[i] = start_postprocess[postprocess_index];
    e_postprocess[i] = end_postprocess[postprocess_index] - start_postprocess[postprocess_index];
    end_postprocess_array[i] = start_preprocess[postprocess_index];

    max_time = MAX(end_preprocess[preprocess_index], MAX(end_infer[inference_index], end_postprocess[postprocess_index]));
    min_time = MIN(end_preprocess[preprocess_index], MIN(end_infer[inference_index], end_postprocess[postprocess_index]));

    e_stall[i] = max_time - min_time;
    execution_time[i] = max_time - start_infer[postprocess_index];
    frame_rate[i] = 1000 / (max_time - start_preprocess[preprocess_index]);

    return;
}


static void *preprocess(void *ptr) 
{

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);

    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset); 
    
    if(barrier_signal) pthread_barrier_wait(&barrier);
#ifdef MEASURE
    start_preprocess[preprocess_index] = get_time_in_ms();
#endif

#ifdef NVTX
    nvtxRangeId_t nvtx_preprocess;
    nvtx_preprocess = nvtxRangeStartA("Preprocess");
#endif

    im = load_image(input, 0, 0, net.c);
    resized = resize_min(im, net.w);
    cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
    X = cropped.data;

#ifdef NVTX
    nvtxRangeEnd(nvtx_preprocess);
#endif

#ifdef MEASURE
    end_preprocess[preprocess_index] = get_time_in_ms();
#endif

}

static void *inference(void *ptr)
{

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(5, &cpuset);

    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset); 

    if(barrier_signal) pthread_barrier_wait(&barrier);

#ifdef MEASURE
    start_infer[inference_index] = get_time_in_ms();
#endif

#ifdef NVTX
        nvtxRangeId_t nvtx_inference;
        nvtx_inference = nvtxRangeStartA("inference");
#endif

    if (device) predictions = network_predict(net, X);
    else predictions = network_predict_cpu(net, X);

#ifdef NVTX
        nvtxRangeEnd(nvtx_inference);
#endif

#ifdef MEASURE
    end_infer[inference_index] = get_time_in_ms();

    // Busy wait for the remaining time
    if (device == 0) remaining_time = 490 - (end_infer[inference_index] - start_infer[inference_index]);
    else remaining_time = 18.91 - (end_infer[inference_index] - start_infer[inference_index]);

    wait_start, wait_end, work_time = 0.0, 0.0, 0.0;
    
    if (remaining_time > 0) {
        wait_start = get_time_in_ms();
        wait_end;
        do {
            wait_end = get_time_in_ms();
            work_time = wait_end - wait_start;
        } while(work_time < remaining_time);
    }

    end_infer_max[inference_index] = get_time_in_ms();
    // printf("end_infer_max : %0.2f \n", end_infer_max[inference_index]);

#endif
}

static void *postprocess(void *ptr)
{

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(9, &cpuset);

    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset); 

    if(barrier_signal) pthread_barrier_wait(&barrier);

#ifdef MEASURE
        start_postprocess[postprocess_index] = get_time_in_ms();
#endif

#ifdef NVTX
        nvtxRangeId_t nvtx_postprocess;
        nvtx_postprocess = nvtxRangeStartA("Postprocess");
#endif
        // __NMS & TOP acccuracy__
        if (object_detection) {
            dets = get_network_boxes(&net, im.w, im.h, demo_thresh, demo_hier_thresh, 0, 1, &nboxes, demo_letterbox);
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }
            draw_detections_v3(im, dets, nboxes, demo_thresh, names, alphabet, l.classes, demo_ext_output);
        } // yolo model
        else {
            if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
            top_k(predictions, net.outputs, top, indexes);
            for(int j = 0; j < top; ++j){
                int index = indexes[j];
                if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
#ifndef MEASURE
                else printf("%s: %f\n",names[index], predictions[index]);
#endif

            }
        } // classifier model

        // __Display__
        // if (!dont_show) {
        //     show_image(im, "predictions");
        //     wait_key_cv(1);
        // }

#ifdef NVTX
        nvtxRangeEnd(nvtx_postprocess);
#endif
    // usleep(15 * 1000);

#ifdef MEASURE
        end_postprocess[postprocess_index] = get_time_in_ms();
#endif
}

void pipeline_jitter(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{

    device = 1; // Choose CPU or GPU

    if (device == 0) printf("\n\nPipeline Architectiure (Jitter Compensation) with \"CPU\"\n");
    else printf("\n\nPipeline Architectiure (Jitter Compensation) with \"GPU\"\n");

    // __CPU AFFINITY SETTING__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); // cpu core index
    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "pthread_setaffinity_np() failed \n");
        exit(0);
    } 

    demo_letterbox = letter_box;

    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_thresh = thresh;
    demo_hier_thresh = hier_thresh;
    demo_ext_output = ext_output;

    options = read_data_cfg(datacfg);
    name_list = option_find_str(options, "names", "data/names.list");
    names_size = 0;
    names = get_labels_custom(name_list, &names_size); //get_labels(name_list)

    buff[256];
    input = buff;

    alphabet = load_alphabet();

    nms = .45;    // 0.4F

    top = 5;
    nboxes = 0;
    indexes = (int*)xcalloc(top, sizeof(int));

    target_model = "yolo";
    object_detection = strstr(cfgfile, target_model);

    net = parse_network_cfg_custom(cfgfile, 1, 1, device); // set batch=1
    l = net.layers[net.n - 1];

    if (weightfile) {
        load_weights(&net, weightfile);
    }
    if (net.letter_box) letter_box = 1;
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    srand(2222222);

    if (filename) strncpy(input, filename, 256);
    else printf("Error! File is not exist.");

    for(int i = 0; i < 3; i++) {
        preprocess_index = i;
        inference_index = i;
        preprocess(&i);
        inference(&i);
    }

    pthread_t preprocess_thread;
    pthread_t inference_thread;
    pthread_t postprocess_thread;

    barrier_signal = 1;

    for (int i = 0; i < num_exp; i++) {

        preprocess_index = thread_index;
        inference_index = (thread_index + 2) % 3;
        postprocess_index = (thread_index + 1) % 3;

#ifdef NVTX
        char task[100];
        sprintf(task, "Task (cpu: %d)", core_id);
        nvtxRangeId_t nvtx_task;
        nvtx_task = nvtxRangeStartA(task);
#endif

#ifndef MEASURE
        printf("\nThread %d is set to CPU core %d\n", core_id, sched_getcpu());
#endif

	    pthread_barrier_init(&barrier, NULL, 3);

        // __Preprocess__
        pthread_create(&preprocess_thread, NULL, preprocess, &preprocess_index);

        // __Inference__
        pthread_create(&inference_thread, NULL, inference, &inference_index);

        // __Postprecess__
        pthread_create(&postprocess_thread, NULL, postprocess, &postprocess_index);
        
        pthread_join(preprocess_thread, NULL);
        pthread_join(inference_thread, NULL);
        pthread_join(postprocess_thread, NULL);

        pthread_barrier_destroy(&barrier);

        // preprocess(&i);
        // inference(&i);
        // postprocess(&i);

        push_data(i);
        thread_index = (thread_index + 1) % 3;

        // free memory
        free_image(im);
        free_image(resized);
        free_image(cropped);

#ifdef NVTX
        nvtxRangeEnd(nvtx_task);
#endif
    }

#ifdef MEASURE
    char file_path[256] = "measure/";

    char* model_name = malloc(strlen(cfgfile) + 1);
    strncpy(model_name, cfgfile + 6, (strlen(cfgfile)-10));
    model_name[strlen(cfgfile)-10] = '\0';

    strcat(file_path, "pipeline/");
    strcat(file_path, model_name);
    strcat(file_path, "/");

    if (device == 0) strcat(file_path, "pipeline_jitter_cpu");
    else strcat(file_path, "pipeline_jitter_gpu");

    strcat(file_path, ".csv");
    if(write_result(file_path) == -1) {
        /* return error */
        exit(0);
    }
#endif

    // free memory
//     free_detections(dets, nboxes);
//     free_ptrs((void**)names, net.layers[net.n - 1].classes);
//     free_list_contents_kvp(options);
//     free_list(options);
//     free_alphabet(alphabet);
    // free_network(net); // Error occur
}
