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

int skip_layers[1000] = {0, };

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

    while (1) {

        printf("Thread %d is set to CPU core %d\n", data.thread_id, sched_getcpu());

        // __Preprocess__
        im = load_image(input, 0, 0, net.c);
        resized = resize_min(im, net.w);
        cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        X = cropped.data;

        time = get_time_point();
        
        // __Inference__
        // if (device) predictions = network_predict(net, X);
        // else predictions = network_predict_cpu(net, X);

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

        cuda_push_array(state.input, net.input_pinned_cpu, size);

        // GPU Inference
        state.workspace = net.workspace;
        for(i = 0; i < gLayer; ++i){
            state.index = i;
            l = net.layers[i];
            if(l.delta_gpu && state.train){
                fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
            }

            l.forward_gpu(l, state);
            if (skip_layers[i]){
                cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
            }
            state.input = l.output_gpu;
        }

        cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
        state.input = l.output;

        // CPU Inference
        state.workspace = net.workspace_cpu;
        gpu_yolo = 0;
        for(i = gLayer; i < net.n; ++i){
            state.index = i;
            l = net.layers[i];
            if(l.delta && state.train && l.train){
                scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
            }
            l.forward(l, state);
            state.input = l.output;
        }

        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

        if (gLayer == net.n) predictions = get_network_output_gpu(net);
        else predictions = get_network_output(net, 0);
        reset_wait_stream_events();
        //cuda_free(state.input);   // will be freed in the free_network()

        printf("\n%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);

        // __Postprecess__
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
            for(i = 0; i < top; ++i){
                index = indexes[i];
                if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
                else printf("%s: %f\n",names[index], predictions[index]);
            }
        }

        // __Display__
        if (!data.dont_show) {
            show_image(im, "predictions");
            wait_key_cv(1);
        }
        // free memory
        free_image(im);
        free_image(resized);
        free_image(cropped);
    }

    // free memory
    free_detections(dets, nboxes);
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_alphabet(alphabet);
    free_network(net);
}


void gpu_accel(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
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

    // pthread_mutex_destroy(&mutex);
    // pthread_cond_destroy(&cond);

    return 0;

}
#else

void gpu_accel(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    printf("!!ERROR!! GPU = 0 \n");
}
#endif  // GPU