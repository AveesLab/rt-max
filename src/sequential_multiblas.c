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

static int coreIDOrder[MAXCORES] = {3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};

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

    double sum_measure_data[num_exp][11];
    for(i = 0; i < num_exp; i++)
    {
        sum_measure_data[i][0] = start_preprocess[i];
        sum_measure_data[i][1] = e_preprocess[i];
        sum_measure_data[i][2] = end_preprocess[i];
        sum_measure_data[i][3] = start_infer[i];
        sum_measure_data[i][4] = e_infer[i];
        sum_measure_data[i][5] = end_infer[i];
        sum_measure_data[i][6] = start_postprocess[i];
        sum_measure_data[i][7] = e_postprocess[i];
        sum_measure_data[i][8] = end_postprocess[i];
        sum_measure_data[i][9] = execution_time[i];
        sum_measure_data[i][10] = frame_rate[i];
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
    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
            "start_preprocess",     "e_preprocess",     "end_preprocess", 
            "start_infer",          "e_infer",          "end_infer", 
            "start_postprocess",    "e_postprocess",    "end_postprocess", 
            "execution_time",       "frame_rate");

    for(i = 0; i < num_exp - startIdx; i++)
    {
        fprintf(fp, "%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f\n",  
                new_sum_measure_data[i][0], new_sum_measure_data[i][1], new_sum_measure_data[i][2], new_sum_measure_data[i][3], 
                new_sum_measure_data[i][4], new_sum_measure_data[i][5], new_sum_measure_data[i][6], new_sum_measure_data[i][7], 
                new_sum_measure_data[i][8], new_sum_measure_data[i][9], new_sum_measure_data[i][10]);
    }
    
    fclose(fp);

    return 1;
}
#endif

void sequential_multiblas(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{

    printf("\n\nSequential-multiblas with %d blas \n", num_blas);

    // __CPU AFFINITY SETTING__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreIDOrder[0], &cpuset); // cpu core index
    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    if (ret != 0) {
        fprintf(stderr, "pthread_setaffinity_np() failed \n");
        exit(0);
    } 

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list)

    char buff[256];
    char *input = buff;

    image **alphabet = load_alphabet();

    float nms = .45;    // 0.4F
    double time;

    int top = 5;
    int nboxes, index, i, j, k = 0;
    int* indexes = (int*)xcalloc(top, sizeof(int));

    image im, resized, cropped;
    float *X, *predictions;
    detection *dets;

    char *target_model = "yolo";
    int object_detection = strstr(cfgfile, target_model);

    int device = 0; // Choose CPU or GPU

    network net = parse_network_cfg_custom(cfgfile, 1, 1, device); // set batch=1
    layer l = net.layers[net.n - 1];

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

    for (i = 0; i < num_exp; i++) {

#ifdef NVTX
        char task[100];
        sprintf(task, "Task (cpu: %d)", coreIDOrder[0]);
        nvtxRangeId_t nvtx_task;
        nvtx_task = nvtxRangeStartA(task);
#endif

#ifndef MEASURE
        printf("\nThread %d is set to CPU core %d\n", coreIDOrder[0], sched_getcpu());
#endif

        time = get_time_in_ms();
        // __Preprocess__
#ifdef MEASURE
        start_preprocess[i] = get_time_in_ms();
#endif

        im = load_image(input, 0, 0, net.c);
        resized = resize_min(im, net.w);
        cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        X = cropped.data;

#ifdef MEASURE
        end_preprocess[i] = get_time_in_ms();
        e_preprocess[i] = end_preprocess[i] - start_preprocess[i];
#endif
        
        // __Inference__
#ifdef MEASURE
        start_infer[i] = get_time_in_ms();
#endif

        openblas_set_num_threads(num_blas);
        CPU_ZERO(&cpuset);
        CPU_SET(coreIDOrder[0], &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

        for(int i = 1; i < num_blas; i++) {
                CPU_ZERO(&cpuset);
                CPU_SET(coreIDOrder[i], &cpuset);
                openblas_setaffinity(i-1, sizeof(cpuset), &cpuset);
        }
        
        if (device) predictions = network_predict(net, X);
        else predictions = network_predict_cpu(net, X);

#ifdef MEASURE
        end_infer[i] = get_time_in_ms();
        e_infer[i] = end_infer[i] - start_infer[i];
#endif

        // __Postprecess__
#ifdef MEASURE
        start_postprocess[i] = get_time_in_ms();
#endif

        // __NMS & TOP acccuracy__
        if (object_detection) {
            dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }
            draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
        } // yolo model
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
        } // classifier model

        // __Display__
        // if (!dont_show) {
        //     show_image(im, "predictions");
        //     wait_key_cv(1);
        // }

#ifdef MEASURE
        end_postprocess[i] = get_time_in_ms();
        e_postprocess[i] = end_postprocess[i] - start_postprocess[i];
        execution_time[i] = end_postprocess[i] - start_preprocess[i];
        frame_rate[i] = 1000.0 / execution_time[i];
        // printf("\n%s: Predicted in %0.3f milli-seconds.\n", input, e_infer[i]);
#else
        execution_time[i] = get_time_in_ms() - time;
        frame_rate[i] = 1000.0 / (execution_time[i] / 1); // 1 single thread
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

#ifdef MEASURE
    char file_path[256] = "measure/";

    char* model_name = malloc(strlen(cfgfile) + 1);
    strncpy(model_name, cfgfile + 6, (strlen(cfgfile)-10));
    model_name[strlen(cfgfile)-10] = '\0';

    char core_idx[10];
    sprintf(core_idx, "%02dblas", num_blas);

    strcat(file_path, "sequential-multiblas/");
    strcat(file_path, model_name);
    strcat(file_path, "/");

    strcat(file_path, "sequential_");
    strcat(file_path, core_idx);

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
