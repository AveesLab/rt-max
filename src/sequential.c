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

#ifdef MEASURE
#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif
#endif


static double execution_time[1000];

/* Timestamp in ms */
double get_time_in_ms(void)
{
    struct timespec time_after_boot;
    clock_gettime(CLOCK_MONOTONIC,&time_after_boot);
    return (time_after_boot.tv_sec*1000+time_after_boot.tv_nsec*0.000001);
}

void sequential(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{

    int device = isGPU; // Choose CPU or GPU

    if (device == 0) printf("\n\nSequential(%dth core) Architectiure with \"CPU\"\n", core_id);
    else printf("\n\nSequential(%dth core) Architectiure with \"GPU\"\n", core_id);

    // __CPU AFFINITY SETTING__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset); // cpu core index
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
    printf("\nThread %d is set to CPU core %d\n", core_id, sched_getcpu());

    for (i = 0; i < num_exp; i++) {
        time = get_time_in_ms();

        im = load_image(input, 0, 0, net.c);
        resized = resize_min(im, net.w);
        cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        X = cropped.data;

        // __Inference__
        if (device) predictions = network_predict(net, X);
        else predictions = network_predict_cpu(net, X);

        // __Postprecess__
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
                // else printf("%s: %f\n",names[index], predictions[index]);
            }
        } // classifier model

        // __Display__
        // if (!dont_show) {
        //     show_image(im, "predictions");
        //     wait_key_cv(1);
        // }

        execution_time[i] = get_time_in_ms() - time;
        printf("\n%s: Predicted in %0.3f milli-seconds.\n", input, execution_time[i]);

        // free memory
        free_image(im);
        free_image(resized);
        free_image(cropped);
    }

    // free memory
//     free_detections(dets, nboxes);
//     free_ptrs((void**)names, net.layers[net.n - 1].classes);
//     free_list_contents_kvp(options);
//     free_list(options);
//     free_alphabet(alphabet);
    // free_network(net); // Error occur
}
