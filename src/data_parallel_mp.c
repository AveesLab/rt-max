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

#ifdef MEASURE
    double start_preprocess[1000];
    double end_preprocess[1000];
    double e_preprocess[1000];

    double start_infer[1000];
    double end_infer[1000];
    double e_infer[1000];

    double start_postprocess[1000];
    double end_postprocess[1000];
    double e_postprocess[1000];

    double execution_time[1000];
    double frame_rate[1000];
#endif

} process_data_t;


#ifdef MEASURE
static int write_result(char *file_path, process_data_t *data) 
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
            "core_id", "", "e_preprocess", "end_preprocess", 
            "start_infer", "e_infer", "end_infer", 
            "start_postprocess", "e_postprocess", "end_postprocess", 
            "execution_time", "frame_rate");

    for(i = 0; i < num_exp * num_process; i++)
    {
        int core_id = (i + 1) - (i / num_process) * num_process;
        int count = i / num_process;
        fprintf(fp, "%d,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f\n",  
                core_id, 
                data[core_id - 1].start_preprocess[count], data[core_id - 1].e_preprocess[count], data[core_id - 1].end_preprocess[count], 
                data[core_id - 1].start_infer[count], data[core_id - 1].e_infer[count], data[core_id - 1].end_infer[count], 
                data[core_id - 1].start_postprocess[count], data[core_id - 1].e_postprocess[count], data[core_id - 1].end_postprocess[count], 
                data[core_id - 1].execution_time[count], data[core_id - 1].frame_rate[count]);
    }
    
    fclose(fp);

    return 1;
}
#endif

#ifdef MEASURE
static void processFunc(process_data_t data, int write_fd)
#else
static void processFunc(process_data_t data)
#endif
{

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
    int nboxes, index, i, j, k = 0;
    int* indexes = (int*)xcalloc(top, sizeof(int));

    image im, resized, cropped;
    float *X, *predictions;
    detection *dets;

    char *target_model = "yolo";
    int object_detection = strstr(data.cfgfile, target_model);

    int device = 0; // Choose CPU or GPU

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
        printf("\nThread %d is set to CPU core %d count(%d) : %d \n\n", data.process_id, sched_getcpu(), data.process_id, i);
#else
        printf("\nThread %d is set to CPU core %d\n\n", data.process_id, sched_getcpu());
#endif
        // __Preprocess__
#ifdef MEASURE
        data.start_preprocess[i] = get_time_in_ms();
#endif

        im = load_image(input, 0, 0, net.c);
        resized = resize_min(im, net.w);
        cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        X = cropped.data;

#ifdef MEASURE
        data.end_preprocess[i] = get_time_in_ms();
        data.e_preprocess[i] = data.end_preprocess[i] - data.start_preprocess[i];
#endif

        // __Inference__
#ifdef MEASURE
        data.start_infer[i] = get_time_in_ms();
#else
        double time = get_time_point();
#endif

        if (device) predictions = network_predict(net, X);
        else predictions = network_predict_cpu(net, X);

#ifdef MEASURE
        data.end_infer[i] = get_time_in_ms();
        data.e_infer[i] = data.end_infer[i] - data.start_infer[i];
        printf("\n%s: Predicted in %0.3f milli-seconds.\n", input, data.e_infer[i]);
#else
        printf("\n%s: Predicted in %0.3f milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);
#endif


        // __Postprecess__
#ifdef MEASURE
        data.start_postprocess[i] = get_time_in_ms();
#endif

        // __NMS & TOP acccuracy__
        if (object_detection) {
            dets = get_network_boxes(&net, im.w, im.h, data.thresh, data.hier_thresh, 0, 1, &nboxes, data.letter_box);
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }
            draw_detections_v3(im, dets, nboxes, data.thresh, names, alphabet, l.classes, data.ext_output);
        } // yolo model
        else {
            if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
            top_k(predictions, net.outputs, top, indexes);
            for(i = 0; i < top; ++i){
                index = indexes[i];
                if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
                else printf("%s: %f\n",names[index], predictions[index]);
            }
        } // classifier model

        // __Display__
        // if (!data.dont_show) {
        //     show_image(im, "predictions");
        //     wait_key_cv(1);
        // }

#ifdef MEASURE
        data.end_postprocess[i] = get_time_in_ms();
        data.e_postprocess[i] = data.end_postprocess[i] - data.start_postprocess[i];
        data.execution_time[i] = data.end_postprocess[i] - data.start_preprocess[i];
        data.frame_rate[i] = 1000.0 / data.execution_time[i];
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
    write(write_fd, &data, sizeof(process_data_t));
#endif

    // free memory
    free_detections(dets, nboxes);
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_alphabet(alphabet);
    // free_network(net); // Error occur
}

void data_parallel_mp(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    int i;

    pid_t pid;
    int status;

    int fd[num_process][2];

    process_data_t data[num_process];

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

        pid = fork();
        if (pid == 0) { // child process

#ifdef MEASURE
            close(fd[i][0]); // close reading end in the child
            processFunc(data[i], fd[i][1]);
            close(fd[i][1]);
#else
            processFunc(data[i]);
#endif

            exit(0);
        } else if (pid < 0) {
            perror("fork");
            exit(1);
        }
    }

#ifdef MEASURE
    process_data_t receivedData[num_process];

    // In the parent process, read data from all child processes
    for (i = 0; i < num_process; i++) {
        close(fd[i][1]); // close writing end in the parent
        read(fd[i][0], &receivedData[i], sizeof(process_data_t));
        data[i] = receivedData[i];
        close(fd[i][0]);
    }
#endif

    for (i = 0; i < num_process; i++) {
        wait(&status);
    }

#ifdef MEASURE
    printf("\n!!Write CSV File!! \n");
    char file_path[256] = "measure/";

    char* model_name = malloc(strlen(cfgfile) + 1);
    strncpy(model_name, cfgfile + 6, (strlen(cfgfile)-10));
    model_name[strlen(cfgfile)-10] = '\0';
    

    strcat(file_path, "data-parallel-mp/");
    strcat(file_path, model_name);
    strcat(file_path, "/");

    strcat(file_path, "data-parallel-mp");

    strcat(file_path, ".csv");
    if(write_result(file_path, data) == -1) {
        /* return error */
        exit(0);
    }
#endif

    return 0;

}
#else

void data_parallel_mp(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    printf("!!ERROR!! MULTI_PROCESSOR = 0 \n");
}
#endif  // MULTI_PROCESSOR