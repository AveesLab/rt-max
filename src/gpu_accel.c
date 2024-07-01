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

#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#define START_INDEX 5
#define END_INDEX 2
#define ACCEPTABLE_JITTER 3
#define NUM_SPLIT 4

pthread_barrier_t barrier;

static int coreIDOrder[MAXCORES] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
static network net_list[MAXCORES];
static pthread_mutex_t mutex_init = PTHREAD_MUTEX_INITIALIZER;
static double start_time[MAXCORES] = {0,0,0,0,0,0,0,0,0,0,0,0};
static int openblas_thread;
static pthread_mutex_t mutex_gpu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

static int current_thread = 1;

static double execution_time_wo_waiting;
char *g_datacfg;
char *g_cfgfile;
char *g_weightfile;
char *g_filename;
float g_thresh;
float g_hier_thresh;
int g_dont_show;
int g_ext_output;
int g_save_labels;
char *g_outfile;
int g_letter_box;
int g_benchmark_layers;
bool isTest;
static int division_count;

int skipped_layers[1000] = {0, };
int skip_layers[1000][10];

list *options;
char *name_list;
static int names_size = 0;
char **names;
char buff[256];
char *input;
image **alphabet;
static float nms = .45;
static int top = 5;
static int nboxes = 0;
static int g_index = 0;

int indexes[5];

detection *dets;

// image im, resized, cropped;
float *predictions;
// float *X;

static char *target_model = "yolo";
int object_detection;

static int device = 1; // Choose CPU or GPU
extern gpu_yolo;

static double core_id_list[1000];
static double start_preprocess[1000];
static double end_preprocess[1000];
static double e_preprocess[1000];
static double e_preprocess_max[1000];

static double start_infer[1000];

static double start_gpu_waiting[1000];
static double waiting_gpu[1000];
static double start_gpu_infer[1000];
static double e_gpu_infer[1000];
static double end_gpu_infer[1000];
static double start_cpu_infer[1000];
static double e_cpu_infer[1000];
static double end_cpu_infer[1000];

static double end_infer[1000];

static double e_gpu_infer_max[1000];
static double e_cpu_infer_max[1000];

static double start_gpu_synchronize[1000];
static double end_gpu_synchronize[1000];
static double e_gpu_synchronize[1000];

static double e_infer[1000];

static double layer_time[400][1000];
static double layer_time_logic[400][1000];
static double max_layer_time[400];

static double start_postprocess[1000];
static double end_postprocess[1000];
static double e_postprocess[1000];

static double check_jitter[1000] = {0, };
static float R;

static double execution_time[1000];
static double execution_time_max[1000];

static double frame_rate[1000];

static float max_preprocess_time;
static float max_gpu_infer_time;
static float max_cpu_infer_time;
static float max_execution_time;
static float release_interval;
static float R;
static int num_network;

int visible_exp;

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
    double sum = 0;
    int total_num_exp = num_exp * num_thread;
    int skip_num_exp =  START_INDEX * num_thread;
    int end_num_exp =  END_INDEX * num_thread;
    int i;
    for(i = skip_num_exp ; i < total_num_exp - end_num_exp; i++) {
        sum += arr[i];
    }
    return (sum / (total_num_exp-skip_num_exp-end_num_exp)) * 1.05;
}

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
static int write_acceleration_info_before_compensation() {
    int startIdx = num_thread * START_INDEX; // Delete some ROWs
    int endIdx = num_thread * END_INDEX; // Delete some ROWs

    char file_path[256] = "measure/";

    char* model_name = malloc(strlen(g_cfgfile) + 1);
    strncpy(model_name, g_cfgfile + 6, (strlen(g_cfgfile)-10));
    model_name[strlen(g_cfgfile)-10] = '\0';
    
    strcat(file_path, "gpu-accel-GC/");

    strcat(file_path, model_name);
    strcat(file_path, "-multithread/");

    char num_threads__[20];
    sprintf(num_threads__, "%dthread/", num_thread);
    strcat(file_path, num_threads__);

    strcat(file_path, "gpu-accel-");

    char gpu_portion[20];
    sprintf(gpu_portion, "%03dglayer", gLayer);
    strcat(file_path, gpu_portion);

    strcat(file_path, "_accel_info_no_compensation");

    strcat(file_path, ".csv");

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

    char layer_id[20];
    for (int i = 0; i < num_network; i++) {
        sprintf(layer_id, "layer[%d]", i);
        fprintf(fp, "%s,", layer_id);
    }
    fprintf(fp, "\n");

    for(int j = 0; j < gLayer; j++) {
        fprintf(fp, "%s,", "GPU");
    }
    for(int j = gLayer; j < num_network; j++) {
        fprintf(fp, "%s,", "CPU");
    }

    fprintf(fp, "\n");

    for(i = 0; i < num_exp * num_thread - startIdx - endIdx; i++)
    {
        for (int j = 0; j < num_network; j++) {
            fprintf(fp, "%0.3f,", layer_time[j][i + startIdx]);
        }
        fprintf(fp, "\n");
    }
    
    return 1;
}

static int write_acceleration_info() {
    int startIdx = num_thread * START_INDEX; // Delete some ROWs
    int endIdx = num_thread * END_INDEX; // Delete some ROWs

    char file_path[256] = "measure/";

    char* model_name = malloc(strlen(g_cfgfile) + 1);
    strncpy(model_name, g_cfgfile + 6, (strlen(g_cfgfile)-10));
    model_name[strlen(g_cfgfile)-10] = '\0';
    
    strcat(file_path, "gpu-accel-GC/");

    strcat(file_path, model_name);
    strcat(file_path, "-multithread/");


    char num_threads__[20];
    sprintf(num_threads__, "%dthread/", num_thread);
    strcat(file_path, num_threads__);

    strcat(file_path, "gpu-accel-");

    char gpu_portion[20];
    sprintf(gpu_portion, "%03dglayer", gLayer);
    strcat(file_path, gpu_portion);

    strcat(file_path, "_accel_info");

    strcat(file_path, ".csv");

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

    char layer_id[20];
    for (int i = 0; i < num_network; i++) {
        sprintf(layer_id, "layer[%d]", i);
        fprintf(fp, "%s,", layer_id);
    }
    fprintf(fp, "\n");

    for(int j = 0; j < gLayer; j++) {
        fprintf(fp, "%s,", "GPU");
    }
    for(int j = gLayer; j < num_network; j++) {
        fprintf(fp, "%s,", "CPU");
    }

    fprintf(fp, "\n");

    for (int j = 0; j < num_network; j++) {
        fprintf(fp, "%0.3f,", max_layer_time[j]);
    }
    fprintf(fp, "\n\n");

    for(i = 0; i < num_exp * num_thread - startIdx - endIdx; i++)
    {
        for (int j = 0; j < num_network; j++) {
            fprintf(fp, "%0.3f,", layer_time_logic[j][i + startIdx]);
        }
        fprintf(fp, "\n");
    }
    
    return 1;
}

static int write_result() 
{
    char file_path[256] = "measure/";

    char* model_name = malloc(strlen(g_cfgfile) + 1);
    strncpy(model_name, g_cfgfile + 6, (strlen(g_cfgfile)-10));
    model_name[strlen(g_cfgfile)-10] = '\0';
    

    strcat(file_path, "gpu-accel-GC/");
    strcat(file_path, model_name);
    strcat(file_path, "-multithread/");

    char num_threads__[20];
    sprintf(num_threads__, "%dthread/", num_thread);
    strcat(file_path, num_threads__);

    strcat(file_path, "gpu-accel-");

    char gpu_portion[20];
    sprintf(gpu_portion, "%03dglayer", gLayer);
    strcat(file_path, gpu_portion);

    strcat(file_path, ".csv");

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

    double sum_measure_data[num_exp * num_thread][33];
    for(i = 0; i < num_exp * num_thread; i++)
    {
        sum_measure_data[i][0] = core_id_list[i];
        sum_measure_data[i][1] = start_preprocess[i];
        sum_measure_data[i][2] = e_preprocess[i];
        sum_measure_data[i][3] = end_preprocess[i];
        sum_measure_data[i][4] = e_preprocess_max[i];
        sum_measure_data[i][5] = max_preprocess_time;
        sum_measure_data[i][6] = start_infer[i]; 
        sum_measure_data[i][7] = end_infer[i];
        sum_measure_data[i][8] = e_infer[i];
        sum_measure_data[i][9] = start_postprocess[i];
        sum_measure_data[i][10] = e_postprocess[i];
        sum_measure_data[i][11] = end_postprocess[i];
        sum_measure_data[i][12] = execution_time[i];
        sum_measure_data[i][13] = execution_time_max[i];
        sum_measure_data[i][14] = max_execution_time;
        sum_measure_data[i][15] = release_interval;
        sum_measure_data[i][16] = 0.0;
        sum_measure_data[i][17] = 0.0;
        sum_measure_data[i][18] = 0.0;
        sum_measure_data[i][19] = 0.0;
        sum_measure_data[i][20] = check_jitter[i];
        sum_measure_data[i][21] = start_gpu_waiting[i];
        sum_measure_data[i][22] = waiting_gpu[i];
        sum_measure_data[i][23] = start_gpu_infer[i];
        sum_measure_data[i][24] = e_gpu_infer[i];
        sum_measure_data[i][25] = end_gpu_infer[i];
        sum_measure_data[i][26] = e_gpu_infer_max[i];
        sum_measure_data[i][27] = max_gpu_infer_time;      
        sum_measure_data[i][28] = start_cpu_infer[i];
        sum_measure_data[i][29] = e_cpu_infer[i];
        sum_measure_data[i][30] = end_cpu_infer[i];
        sum_measure_data[i][31] = e_cpu_infer_max[i];  
        sum_measure_data[i][32] = max_cpu_infer_time;  
    }


    qsort(sum_measure_data, sizeof(sum_measure_data)/sizeof(sum_measure_data[0]), sizeof(sum_measure_data[0]), compare);

    int startIdx = num_thread * START_INDEX; // Delete some ROWs
    int endIdx = num_thread * END_INDEX; // Delete some ROWs
    double new_sum_measure_data[sizeof(sum_measure_data)/sizeof(sum_measure_data[0])-startIdx-endIdx][sizeof(sum_measure_data[0])];

    int newIndex = 0;
    for (int i = startIdx; i < sizeof(sum_measure_data)/sizeof(sum_measure_data[0])-endIdx; i++) {
        for (int j = 0; j < sizeof(sum_measure_data[0]); j++) {
            new_sum_measure_data[newIndex][j] = sum_measure_data[i][j];
        }
        newIndex++;
    }

    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", 
            "core_id", 
            "start_preprocess", "e_preprocess", "end_preprocess", "e_preprocess_max", "e_preprocess_max_value",
            "start_infer", "end_infer", 
            "e_infer",
            "start_postprocess", "e_postprocess", "end_postprocess", 
            "execution_time", "execution_time_max", "max_execution_time", "release_interval",
            "cycle_time", "frame_rate",
            "num_thread", "R", "check_jitter",
            "start_gpu_waiting", "waiting_gpu", 
            "start_gpu_infer", "e_gpu_infer", "end_gpu_infer", "e_gpu_infer_max", "e_gpu_infer_max_value",
            "start_cpu_infer", "e_cpu_infer", "end_cpu_infer", "e_cpu_infer_max", "e_cpu_infer_max_value");

    double frame_rate = 0.0;
    double cycle_time = 0.0;

    for(i = 0; i < num_exp * num_thread - startIdx-endIdx; i++)
    {
        if (new_sum_measure_data[i][19] > ACCEPTABLE_JITTER) continue;

        if (i == 0) cycle_time = NAN;
        else cycle_time = new_sum_measure_data[i][11] - new_sum_measure_data[i-1][11];

        if (i == 0) frame_rate = NAN;
        else frame_rate = 1000/cycle_time;

        new_sum_measure_data[i][16] = cycle_time;
        new_sum_measure_data[i][17] = frame_rate;
        new_sum_measure_data[i][18] = (double)num_thread;
        new_sum_measure_data[i][19] = R;

        fprintf(fp, "%0.0f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f\n",  
                new_sum_measure_data[i][0], new_sum_measure_data[i][1], new_sum_measure_data[i][2], new_sum_measure_data[i][3], 
                new_sum_measure_data[i][4], new_sum_measure_data[i][5], new_sum_measure_data[i][6], new_sum_measure_data[i][7], 
                new_sum_measure_data[i][8], new_sum_measure_data[i][9], new_sum_measure_data[i][10], new_sum_measure_data[i][11], 
                new_sum_measure_data[i][12], new_sum_measure_data[i][13], new_sum_measure_data[i][14], new_sum_measure_data[i][15],
                new_sum_measure_data[i][16], new_sum_measure_data[i][17], new_sum_measure_data[i][18], new_sum_measure_data[i][19],
                new_sum_measure_data[i][20], new_sum_measure_data[i][21], new_sum_measure_data[i][22], new_sum_measure_data[i][23], 
                new_sum_measure_data[i][24], new_sum_measure_data[i][25], new_sum_measure_data[i][26], new_sum_measure_data[i][27], 
                new_sum_measure_data[i][28], new_sum_measure_data[i][29], new_sum_measure_data[i][30], new_sum_measure_data[i][31],
                new_sum_measure_data[i][32]);
    }

    fclose(fp);

    return 1;
}

static void InitNetwork(int threads_id)
{
    network net_init = parse_network_cfg_custom(g_cfgfile, 1, 1, device); // set batch=1

    if (g_weightfile) {
        load_weights(&net_init, g_weightfile);
    }

    if (net_init.letter_box) g_letter_box = 1;
    net_init.benchmark_layers = g_benchmark_layers;
    fuse_conv_batchnorm(net_init);
    calculate_binary_weights(net_init);

    net_list[threads_id] = net_init;
    num_network = net_init.n; 
}

static void SetAffinity(int thread_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreIDOrder[thread_id], &cpuset); // cpu core index
    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "pthread_setaffinity_np() failed \n");
        exit(0);
    } 
}

static void cpu_inference(network_state *state, network *net, layer *l, int count, int thread_id)
{
    start_cpu_infer[count] = get_time_in_ms();
    for(int j = gLayer; j < num_network; j++) {
        double layer_start = get_time_in_ms();
        state->index = j;
        l = &(net->layers[j]);
        l->do_reclaiming = 0;
        if(l->delta && state->train && l->train){
            scal_cpu(l->outputs * l->batch, 0, l->delta, 1);
        }
        l->forward(*l, *state);
        
        state->input = l->output;
        if(j == net->n - 1) {
            predictions = get_network_output(*net, 0);
        }
        layer_time[j][count] = get_time_in_ms() - layer_start;
        layer_time_logic[j][count] = get_time_in_ms() - layer_start;

        if(!isTest && layer_time_logic[j][count] < max_layer_time[j]) {// max_layer_time[j] 저장
            while(layer_time_logic[j][count] < max_layer_time[j]) {
                layer_time_logic[j][count] = get_time_in_ms() - layer_start;
            }
        }
        layer_time_logic[j][count] = get_time_in_ms() - layer_start;
    }
    end_cpu_infer[count] = get_time_in_ms();
    // e_cpu_infer_max[count] = end_cpu_infer[count] - start_cpu_infer[count];
    // if(!isTest && e_cpu_infer_max[count] < max_cpu_infer_time) {// max_layer_time[j] 저장
    //     while(e_cpu_infer_max[count] < max_cpu_infer_time) {
    //         e_cpu_infer_max[count] = get_time_in_ms() - start_cpu_infer[count];
    //     }
    // }
}

static void gpu_inference(network_state *state, network *net, layer *l, int count, int thread_id, float *X)
{
    if (net->gpu_index != cuda_get_device())
        cuda_set_device(net->gpu_index);
    int size = get_network_input_size(*net) * net->batch;
    state->index = 0;
    state->net = *net;
    state->truth = 0;
    state->train = 0;
    state->delta = 0;
    
    if(gLayer != 0) {
        state->input = net->input_state_gpu;
        cuda_push_array(state->input, X, size);
    }
    else {
        state->input = X;
    }
    
    state->workspace = net->workspace;
    start_gpu_waiting[count] = get_time_in_ms();

    pthread_mutex_lock(&mutex_gpu);

    while(current_thread != thread_id) {
        pthread_cond_wait(&cond, &mutex_gpu);
    }
    
    start_gpu_infer[count] = get_time_in_ms();

    for(int j = 0; j < gLayer; j++) {
        double layer_start = get_time_in_ms();
        state->index = j;
        l = &(net->layers[j]);
        if(l->delta_gpu && state->train){
            fill_ongpu(l->outputs * l->batch, 0, l->delta_gpu, 1);
        }

        l->forward_gpu(*l, *state);

        state->input = l->output_gpu;
        if(j == net->n - 1) {
            predictions = get_network_output_gpu(*net);
        }
        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
        layer_time[j][count] = get_time_in_ms() - layer_start; //practice
        layer_time_logic[j][count] = get_time_in_ms() - layer_start;

        if(!isTest && (layer_time_logic[j][count] < max_layer_time[j])) { // max_layer_time[j] 저장
            while(layer_time_logic[j][count] < max_layer_time[j]) {
                layer_time_logic[j][count] = get_time_in_ms() - layer_start;
            }
        }
        layer_time_logic[j][count] = get_time_in_ms() - layer_start;
    }

    end_gpu_infer[count] = get_time_in_ms();
    e_gpu_infer_max[count] = end_gpu_infer[count] - start_gpu_infer[count];
    // if(!isTest && e_gpu_infer_max[count] < max_gpu_infer_time) {// max_layer_time[j] 저장
    //     while(e_gpu_infer_max[count] < max_gpu_infer_time) {
    //         e_gpu_infer_max[count] = get_time_in_ms() - start_gpu_infer[count];
    //     }
    // }
    current_thread = (current_thread) % num_thread + 1;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mutex_gpu);
}

static void preprocess(image *im, image *resized, image *cropped, float *X, network net)
{
    int size = get_network_input_size(net) * net.batch;
    *im = load_image(g_filename, 0, 0, net.c);
    *resized = resize_min(*im, net.w);
    *cropped = crop_image(*resized, (resized->w - net.w)/2, (resized->h - net.h)/2, net.w, net.h);
}

static void postprocess(network net, image im, layer l, int thread_id, int exp_count, int count)
{
    start_postprocess[count] = get_time_in_ms();

    if(object_detection) {
        dets = get_network_boxes(&net, im.w, im.h, g_thresh,g_hier_thresh, 0, 1, &nboxes, g_letter_box);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        draw_detections_v3(im, dets, nboxes, g_thresh, names, alphabet, l.classes, g_ext_output);
    }
    else {
        if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
        top_k(predictions, net.outputs, top, indexes);
        for(int j = 0; j < top; ++j){
            g_index = indexes[j];
            if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",g_index, names[g_index], predictions[g_index], (net.hierarchy->parent[g_index] >= 0) ? names[net.hierarchy->parent[g_index]] : "Root");
            else if (show_accuracy && thread_id == 1 && exp_count == 3)printf("%s: %f\n",names[g_index], predictions[g_index]);
        }
    }
   
    // __Measure Result__

    core_id_list[count] = (double)sched_getcpu();
    
    waiting_gpu[count] = start_gpu_infer[count] - start_gpu_waiting[count];
    e_gpu_infer[count] = end_gpu_infer[count] - start_gpu_infer[count];
    e_cpu_infer[count] = end_cpu_infer[count] - start_cpu_infer[count];
    e_infer[count] = end_infer[count] - start_infer[count];
    
    end_postprocess[count] = get_time_in_ms();
    e_postprocess[count] = end_postprocess[count] - start_postprocess[count];
    execution_time[count] = end_postprocess[count] - start_preprocess[count];
    execution_time_max[count] = end_postprocess[count] - start_preprocess[count];

    if(!isTest && execution_time_max[count] < max_execution_time) {// max_layer_time[j] 저장
        while(execution_time_max[count] < max_execution_time) {
            execution_time_max[count] = get_time_in_ms() - start_preprocess[count];
        }
    } 
    execution_time_max[count] = end_postprocess[count] - start_preprocess[count];
}

static void initThread(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    g_datacfg = datacfg;
    g_cfgfile = cfgfile;
    g_weightfile = weightfile;
    g_filename = filename;
    g_thresh = thresh;
    g_hier_thresh = hier_thresh;
    g_dont_show = dont_show;
    g_ext_output = ext_output;
    g_save_labels = save_labels;
    g_outfile = outfile;
    g_letter_box = letter_box;
    g_benchmark_layers = benchmark_layers;

    R = 0.0;
    max_preprocess_time = 0.0;
    max_gpu_infer_time = 0.0;
    max_cpu_infer_time = 0.0;
    release_interval = 0.0;

    options = read_data_cfg(g_datacfg);
    name_list = option_find_str(options, "names", "data/names.list");
    names_size = 0;
    names = get_labels_custom(name_list, &names_size); //get_labels(name_list)
    input = buff;

    alphabet = load_alphabet();
    object_detection = strstr(g_cfgfile, target_model);

    isTest = true;

    if (g_filename) strncpy(input, g_filename, 256);
    else printf("Error! File is not exist.");

    visible_exp = show_result;

    cpu_set_t cpuset;

    openblas_thread = (MAXCORES - 1) - num_thread + 1;
    openblas_set_num_threads(openblas_thread);
    for (int k = 0; k < openblas_thread - 1; k++) {
        CPU_ZERO(&cpuset);
        CPU_SET(coreIDOrder[(MAXCORES - 1) - k], &cpuset);
        openblas_setaffinity(k, sizeof(cpuset), &cpuset);
    }

    if (visible_exp) printf("\nGPU-accel with %d threads with %d gpu-layer\n", num_thread, gLayer);

}

static void CalcMaxTime(int num_network)
{
    float max_preprocess_print = average(e_preprocess);
    float max_execution_print = average(execution_time);
    float max_gpu_infer_print = average(e_gpu_infer);
    float max_cpu_infer_print = average(e_cpu_infer);
    execution_time_wo_waiting = max_preprocess_print + max_gpu_infer_print + max_cpu_infer_print;
    R = maxOfThree(max_gpu_infer_print, execution_time_wo_waiting/num_thread, 0);
    release_interval = R * num_thread;
    max_preprocess_time = average(e_preprocess);
    max_execution_time = release_interval;
    max_gpu_infer_time = average(e_gpu_infer);
    max_cpu_infer_time = average(e_cpu_infer);

    // if(isTest) {
        for(int h = 0; h < num_network; h++) {	
            double sum = 0;
            for(int k = num_thread * START_INDEX; k < num_thread * (num_exp - END_INDEX); k++) {
                sum += layer_time[h][k];
                division_count += 1;
            }
            sum /= (float)division_count;
            max_layer_time[h] = sum * 1.03;
            division_count = 0;
        }
    // }

    if (visible_exp) {
        printf("e_pre : %0.02f, e_infer_cpu : %0.02f, e_infer_gpu : %0.02f, execution_time : %0.02f, TOTAL/N: %0.02f, Release interval: %0.02f\n", max_preprocess_print, max_cpu_infer_print, max_gpu_infer_print, execution_time_wo_waiting, execution_time_wo_waiting/num_thread, release_interval);
    }
}

static void WriteResult()
{
    if(write_result() == -1) {
        /* return error */
        exit(0);
    }

    if(write_acceleration_info() == -1) {
        exit(0);
    }

    if(write_acceleration_info_before_compensation() == -1) {
        exit(0);
    }
}

#ifdef GPU
static void threadFunc(int arg)
{
    int thread_id = arg;

    // __CPU AFFINITY SETTING__
    SetAffinity(thread_id);

    // __GPU SETUP__
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
        CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    }

    network net = net_list[thread_id];
    layer l;

    srand(2222222);
    double remaining_time = 0.0;

    for (int i = 0; i < num_exp; i++) {
        int count = i * num_thread + thread_id - 1;

        // // __Time Sync__
        if(!isTest) {
            if(i < START_INDEX) {
		        pthread_barrier_wait(&barrier);
                usleep(R * thread_id * 1000);
            }
        }
        else{
            pthread_barrier_wait(&barrier);
        }

        // __Preprocess__
        start_preprocess[count]=get_time_in_ms();
        image im, resized, cropped;
        float *X;
        preprocess(&im, &resized, &cropped, X, net);
        X = cropped.data;

        end_preprocess[count] = get_time_in_ms();
        e_preprocess[count] = end_preprocess[count] - start_preprocess[count];
        e_preprocess_max[count] = get_time_in_ms() - start_preprocess[count];

        if(!isTest && e_preprocess_max[count] < max_preprocess_time) {// max_layer_time[j] 저장
            while(e_preprocess_max[count] < max_preprocess_time) {
                e_preprocess_max[count] = get_time_in_ms() - start_preprocess[count];
            }
        }
        e_preprocess_max[count] = get_time_in_ms() - start_preprocess[count];

        start_infer[count] = get_time_in_ms();

        // GPU Inference
        network_state state;

        gpu_inference(&state, &net, &l, count, thread_id, X);
        cpu_inference(&state, &net, &l, count, thread_id);

        end_infer[count] = get_time_in_ms();

        postprocess(net, im, l, thread_id, i, count);
    }
    pthread_exit(NULL);
}

void gpu_accel(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    initThread(datacfg, cfgfile, weightfile, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers);
        
    pthread_t threads[num_thread];
    int rc;
    int i;

    // if (num_thread < (MAXCORES-1)/*&& (rLayer > 0)*/ ) {
        for(int q = 0; q < 50; q++) {
        // =====================RECLAMING=====================
            if (visible_exp) printf("\n::EXP-%d:: GPU-accel with %d threads with %d gpu-layer [R : %.2f]\n", q, num_thread, gLayer, R);

            reset_check_jitter();
            pthread_barrier_init(&barrier, NULL, num_thread);

            for (i = 0; i < num_thread; i++) {
                int threads_id = i + 1;
                if(isTest) {
                    InitNetwork(threads_id);            
                }
                rc = pthread_create(&threads[i], NULL, threadFunc, threads_id);
                if (rc) {
                    printf("Error: Unable to create thread, %d\n", rc);
                    exit(-1);
                }
            }

            for (i = 0; i < num_thread; i++) {
                pthread_join(threads[i], NULL);
            }
            CalcMaxTime(num_network);

            if(q == 0) {
                isTest = false;
            }
        }
    // }
    pthread_barrier_destroy(&barrier);
    WriteResult();
   
    return 0;

}
#else

void gpu_accel(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    printf("!!ERROR!! GPU = 0 \n");
}
#endif  // GPU
