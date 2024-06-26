// #include <stdlib.h>
// #include <math.h>
// #include "darknet.h"
// #include "network.h"
// #include "parser.h"
// #include "detector.h"
// #include "option_list.h"

// #include <pthread.h>
// #define _GNU_SOURCE
// #include <sched.h>
// #include <unistd.h>

// #ifdef WIN32
// #include <time.h>
// #include "gettimeofday.h"
// #else
// #include <sys/time.h>
// #endif

// #define START_INDEX 5
// #define END_INDEX 2
// #define ACCEPTABLE_JITTER 3
// #define NUM_SPLIT 4

// pthread_barrier_t barrier;

// char inference_order[NUM_SPLIT][20] = {}; // "GPU", "Reclaiming" "CPU"
// int infer_start[NUM_SPLIT] = {0, };
// int infer_end[NUM_SPLIT] = {0, };

// static int coreIDOrder[MAXCORES] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
// // static int coreIDOrder[MAXCORES] = {0,1,2,3,4,5,6,7,8,9,10,11};
// static network net_list[MAXCORES];
// static pthread_mutex_t mutex_init = PTHREAD_MUTEX_INITIALIZER;
// static double start_time[MAXCORES] = {0,0,0,0,0,0,0,0,0,0,0,0};
// static int openblas_thread;
// static pthread_mutex_t mutex_gpu = PTHREAD_MUTEX_INITIALIZER;
// static pthread_mutex_t mutex_reclaim = PTHREAD_MUTEX_INITIALIZER;
// static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
// static pthread_cond_t cond2 = PTHREAD_COND_INITIALIZER;

// static int current_thread = 1;
// static int current_thread2 = 1;

// static double execution_time_wo_waiting;
// char *g_datacfg;
// char *g_cfgfile;
// char *g_weightfile;
// char *g_filename;
// float g_thresh;
// float g_hier_thresh;
// int g_dont_show;
// int g_ext_output;
// int g_save_labels;
// char *g_outfile;
// int g_letter_box;
// int g_benchmark_layers;
// bool isTest;
// bool isSet;
// bool isReclaiming;
// static int division_count;

// static int skipped_layers[1000] = {0, };

// list *options;
// char *name_list;
// int names_size = 0;
// char **names;
// char buff[256];
// char *input;
// image **alphabet;
// float nms = .45;
// int top = 5;
// int nboxes = 0;
// int g_index = 0;

// int indexes[5];

// detection *dets;

// image im, resized, cropped;
// float *X, *predictions;

// char *target_model = "yolo";
// int object_detection;

// int device = 1; // Choose CPU or GPU
// extern gpu_yolo;

// static double core_id_list[1000];
// static double start_preprocess[1000];
// static double end_preprocess[1000];
// static double e_preprocess[1000];
// static double e_preprocess_max[1000];

// static double start_infer[1000];

// static double start_gpu_waiting[4][1000];
// static double waiting_gpu[4][1000];
// static double start_gpu_infer[4][1000];
// static double e_gpu_infer[4][1000];
// static double end_gpu_infer[4][1000];
// static double start_reclaim_waiting[4][1000];
// static double waiting_reclaim[4][1000];
// static double start_reclaim_infer[4][1000];
// static double e_reclaim_infer[4][1000];
// static double end_reclaim_infer[4][1000];
// static double start_cpu_infer[4][1000];
// static double e_cpu_infer[4][1000];
// static double end_cpu_infer[4][1000];

// static double end_infer[1000];

// static double e_gpu_infer_max[4][1000];

// static double start_gpu_synchronize[1000];
// static double end_gpu_synchronize[1000];
// static double e_gpu_synchronize[1000];

// static double e_infer[1000];

// static double layer_time[400][1000];
// static double max_layer_time[400];

// static double start_postprocess[1000];
// static double end_postprocess[1000];
// static double e_postprocess[1000];

// static double check_jitter[1000] = {0, };
// static float R;

// static double execution_time[1000];
// static double execution_time_max[1000];

// static double frame_rate[1000];

// static float max_preprocess_time;
// static float max_gpu_infer_time;
// static float max_reclaim_infer_time;
// static float max_cpu_infer_time;
// static float release_interval;
// static float R;
// static int num_network;

// static int reset_check_jitter() {
//     for (int check_num = 0; check_num < 1000; check_num++) {
//         check_jitter[check_num] = 0;
//     }
//     return 1;
// }

// static int is_GPU_larger(double a, double b) {
//     return (a - b) >= 2 ? 1 : 0; // Check 2ms differnce
// }

// static double average(double arr[]){
//     double sum;
//     int total_num_exp = num_exp * num_thread;
//     int skip_num_exp =  START_INDEX * num_thread;
//     int end_num_exp =  END_INDEX * num_thread;
//     int i;
//     for(i = skip_num_exp ; i < total_num_exp - end_num_exp; i++) {
//         sum += arr[i];
//     }
//     return (sum / (total_num_exp-skip_num_exp-end_num_exp)) * 1.05;
// }

// static int compare(const void *a, const void *b) {
//     double valueA = *((double *)a + 1);
//     double valueB = *((double *)b + 1);

//     if (valueA < valueB) return -1;
//     if (valueA > valueB) return 1;
//     return 0;
// }

// static double maxOfThree(double a, double b, double c) {
//     double max = a;

//     if (b > max) {
//         max = b;
//     }
//     if (c > max) { 
//         max = c;
//     }
//     return max; 
// }

// static int write_acceleration_info() {
//     int startIdx = num_thread * START_INDEX; // Delete some ROWs
//     int endIdx = num_thread * END_INDEX; // Delete some ROWs

//     char file_path[256] = "measure/";

//     char* model_name = malloc(strlen(g_cfgfile) + 1);
//     strncpy(model_name, g_cfgfile + 6, (strlen(g_cfgfile)-10));
//     model_name[strlen(g_cfgfile)-10] = '\0';
    
//     strcat(file_path, "cpu-reclaiming-GRC/");

//     strcat(file_path, model_name);
//     strcat(file_path, "-multithread/");


//     char num_threads__[20];
//     sprintf(num_threads__, "%dthread/", num_thread);
//     strcat(file_path, num_threads__);

//     char gpu_portion[20];
//     sprintf(gpu_portion, "%dglayer/", infer_end[0]);
//     strcat(file_path, gpu_portion);

//     strcat(file_path, "cpu-reclaiming_");

//     char rec_portion[20];
//     sprintf(rec_portion, "%03drlayer", infer_end[1]);
//     strcat(file_path, rec_portion);

//     char splitnum[20];
//     sprintf(splitnum, "_%03dsplit", NUM_SPLIT);
//     strcat(file_path, splitnum);

//     strcat(file_path, "_accel_info");

//     strcat(file_path, ".csv");

//     static int exist=0;
//     FILE *fp;
//     int tick = 0;

//     fp = fopen(file_path, "w+");

//     int i;
//     if (fp == NULL) 
//     {
//         /* make directory */
//         while(!exist)
//         {
//             int result;

//             usleep(10 * 1000);

//             result = mkdir(MEASUREMENT_PATH, 0766);
//             if(result == 0) { 
//                 exist = 1;

//                 fp = fopen(file_path,"w+");
//             }

//             if(tick == 100)
//             {
//                 fprintf(stderr, "\nERROR: Fail to Create %s\n", file_path);

//                 return -1;
//             }
//             else tick++;
//         }
//     }
//     else printf("Write output in %s\n", file_path); 

//     char layer_id[20];
//     for (int i = 0; i < num_network; i++) {
//         sprintf(layer_id, "layer[%d]", i);
//         fprintf(fp, "%s,", layer_id);
//     }
//     fprintf(fp, "\n");

//     for (int j = 0; j < num_network; j++) {
//         for(int i = 0; i < NUM_SPLIT; i++) {
//             if(j < infer_end[i]) {
//                 fprintf(fp, "%s,", inference_order[i]);
//                 break;
//             }
//         }
//     }
//     fprintf(fp, "\n");

//     for(i = 0; i < num_exp * num_thread - startIdx - endIdx; i++)
//     {
//         for (int j = 0; j < num_network; j++) {
//             fprintf(fp, "%0.3f,", layer_time[j][i + startIdx]);
//         }
//         fprintf(fp, "\n");
//     }
    
//     return 1;
// }

// static int write_result_reclaiming() 
// {
//     char file_path[256] = "measure/";

//     char* model_name = malloc(strlen(g_cfgfile) + 1);
//     strncpy(model_name, g_cfgfile + 6, (strlen(g_cfgfile)-10));
//     model_name[strlen(g_cfgfile)-10] = '\0';
    

//     strcat(file_path, "cpu-reclaiming-GRC/");
//     strcat(file_path, model_name);
//     strcat(file_path, "-multithread/");

//     char num_threads__[20];
//     sprintf(num_threads__, "%dthread/", num_thread);
//     strcat(file_path, num_threads__);

//     char gpu_portion[20];
//     sprintf(gpu_portion, "%dglayer/", infer_end[0]);
//     strcat(file_path, gpu_portion);

//     strcat(file_path, "cpu-reclaiming_");

//     char reclaim_portion[20];
//     sprintf(reclaim_portion, "%03drlayer", infer_end[1]);
//     strcat(file_path, reclaim_portion);

//     char splitnum[20];
//     sprintf(splitnum, "_%03dsplit", NUM_SPLIT);
//     strcat(file_path, splitnum);

//     strcat(file_path, ".csv");

//     static int exist=0;
//     FILE *fp;
//     int tick = 0;

//     fp = fopen(file_path, "w+");

//     int i;
//     if (fp == NULL) 
//     {
//         /* make directory */
//         while(!exist)
//         {
//             int result;

//             usleep(10 * 1000);

//             result = mkdir(MEASUREMENT_PATH, 0766);
//             if(result == 0) { 
//                 exist = 1;

//                 fp = fopen(file_path,"w+");
//             }

//             if(tick == 100)
//             {
//                 fprintf(stderr, "\nERROR: Fail to Create %s\n", file_path);

//                 return -1;
//             }
//             else tick++;
//         }
//     }
//     else printf("Write output in %s\n", file_path); 

//     int data_count = 0;
//     int write_index = 0;

//     for(i = 0; i < NUM_SPLIT; i++) {
//         if(strcmp(inference_order[i], "GPU\0") == 0) {
//             data_count += 7;
//         }
//         else if(strcmp(inference_order[i], "CPU\0") == 0) {
//             data_count += 4;
//         }
//         else if(strcmp(inference_order[i], "Reclaiming") == 0) {
//             data_count += 6;
//         }
//     }
//     double sum_measure_data[num_exp * num_thread][20 + data_count];
//     for(i = 0; i < num_exp * num_thread; i++)
//     {
//         sum_measure_data[i][0] = core_id_list[i];
//         sum_measure_data[i][1] = start_preprocess[i];
//         sum_measure_data[i][2] = e_preprocess[i];
//         sum_measure_data[i][3] = end_preprocess[i];
//         sum_measure_data[i][4] = e_preprocess_max[i];
//         sum_measure_data[i][5] = max_preprocess_time;
//         sum_measure_data[i][6] = start_infer[i]; 
//         sum_measure_data[i][7] = end_infer[i];
//         sum_measure_data[i][8] = e_infer[i];
//         sum_measure_data[i][9] = start_postprocess[i];
//         sum_measure_data[i][10] = e_postprocess[i];
//         sum_measure_data[i][11] = end_postprocess[i];
//         sum_measure_data[i][12] = execution_time[i];
//         sum_measure_data[i][13] = execution_time_max[i];
//         sum_measure_data[i][14] = release_interval;
//         sum_measure_data[i][15] = 0.0;
//         sum_measure_data[i][16] = 0.0;
//         sum_measure_data[i][17] = 0.0;
//         sum_measure_data[i][18] = 0.0;
//         sum_measure_data[i][19] = check_jitter[i];
//     }

//     for(int j = 0; j < data_count; j += write_index) {
//         static int k = 0;
//         if(strcmp(inference_order[k], "GPU\0") == 0) {
//             for(i = 0; i < num_exp * num_thread; i++) {
//                 sum_measure_data[i][20 + j] = start_gpu_waiting[k][i];
//                 sum_measure_data[i][21 + j] = waiting_gpu[k][i];
//                 sum_measure_data[i][22 + j] = start_gpu_infer[k][i];
//                 sum_measure_data[i][23 + j] = e_gpu_infer[k][i];
//                 sum_measure_data[i][24 + j] = end_gpu_infer[k][i];
//                 sum_measure_data[i][25 + j] = e_gpu_infer_max[k][i];
//                 sum_measure_data[i][26 + j] = max_gpu_infer_time;
//             }  
//             write_index = 7;
//             k++;
//         }
//         else if(strcmp(inference_order[k], "CPU\0") == 0) {
//             for(i = 0; i < num_exp * num_thread; i++) {
//                 sum_measure_data[i][20 + j] = start_cpu_infer[k][i];
//                 sum_measure_data[i][21 + j] = e_cpu_infer[k][i];
//                 sum_measure_data[i][22 + j] = end_cpu_infer[k][i];
//                 sum_measure_data[i][23 + j] = max_cpu_infer_time;
//             }
//             write_index = 4;
//             k++;
//         }
//         else if(strcmp(inference_order[k], "Reclaiming") == 0) {
//             for(i = 0; i < num_exp * num_thread; i++) {
//                 sum_measure_data[i][20 + j] = start_reclaim_waiting[k][i];
//                 sum_measure_data[i][21 + j] = waiting_reclaim[k][i];
//                 sum_measure_data[i][22 + j] = start_reclaim_infer[k][i];
//                 sum_measure_data[i][23 + j] = e_reclaim_infer[k][i];
//                 sum_measure_data[i][24 + j] = end_reclaim_infer[k][i];
//                 sum_measure_data[i][25 + j] = max_reclaim_infer_time;
//             }
//             write_index = 6;
//             k++;
//         }
        
//     }

//     qsort(sum_measure_data, sizeof(sum_measure_data)/sizeof(sum_measure_data[0]), sizeof(sum_measure_data[0]), compare);

//     int startIdx = num_thread * START_INDEX; // Delete some ROWs
//     int endIdx = num_thread * END_INDEX; // Delete some ROWs
//     double new_sum_measure_data[sizeof(sum_measure_data)/sizeof(sum_measure_data[0])-startIdx-endIdx][sizeof(sum_measure_data[0])];

//     int newIndex = 0;
//     for (int i = startIdx; i < sizeof(sum_measure_data)/sizeof(sum_measure_data[0])-endIdx; i++) {
//         for (int j = 0; j < sizeof(sum_measure_data[0]); j++) {
//             new_sum_measure_data[newIndex][j] = sum_measure_data[i][j];
//         }
//         newIndex++;
//     }

//     fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,", 
//             "core_id", 
//             "start_preprocess", "e_preprocess", "end_preprocess", "e_preprocess_max", "e_preprocess_max_value",
//             "start_infer", "end_infer", 
//             "e_infer",
//             "start_postprocess", "e_postprocess", "end_postprocess", 
//             "execution_time", "execution_time_max", "release_interval",
//             "cycle_time", "frame_rate",
//             "num_thread", "R", "check_jitter");

//     char *gpu_headers[] = {"start_gpu_waiting", "waiting_gpu", 
//         "start_gpu_infer", "e_gpu_infer", "end_gpu_infer", "e_gpu_infer_max", "e_gpu_infer_max_value"};

//     char *rec_headers[] = {"start_reclaim_waiting", "waiting_reclaim", 
//         "start_reclaim_infer", "e_reclaim_infer", "end_reclaim_infer", "e_reclaim_infer_max_value"}; 
    
//     char *cpu_headers[] = {"start_cpu_infer", "e_cpu_infer", "end_cpu_infer", "e_cpu_infer_max_value"};
    
//     write_index = 0;
//     for(int j = 0; j < data_count; j += write_index) {
//         static int h = 0;
//         if(strcmp(inference_order[h], "GPU\0") == 0) {
//             static gpu_count = 0;
//             for(int k = 0; k < 7; ++k) {
//                 char dynamic_header[100];  // Buffer to hold the new header with number
//                 sprintf(dynamic_header, "%s_%d", gpu_headers[k], gpu_count + 1);  // Create new header with number
//                 fprintf(fp, "%s", dynamic_header);
//                 fprintf(fp, ",");
//             }
//             write_index = 7;
//             gpu_count ++;
//             h ++;
//         }
//         else if(strcmp(inference_order[h], "CPU\0") == 0) {
//             static cpu_count = 0;
//             for(int k = 0; k < 4; ++k) {
//                 char dynamic_header[100];  // Buffer to hold the new header with number
//                 sprintf(dynamic_header, "%s_%d", cpu_headers[k], cpu_count + 1);  // Create new header with number
//                 fprintf(fp, "%s", dynamic_header);
//                 fprintf(fp, ",");
//             }
//             write_index = 4;
//             cpu_count ++;
//             h ++;
//         }
//         else if(strcmp(inference_order[h], "Reclaiming") == 0) {
//             static rec_count = 0;
//             for(int k = 0; k < 6; ++k) {
//                 char dynamic_header[100];  // Buffer to hold the new header with number
//                 sprintf(dynamic_header, "%s_%d", rec_headers[k], rec_count + 1);  // Create new header with number
//                 fprintf(fp, "%s", dynamic_header);
//                 fprintf(fp, ",");
//             }
//             write_index = 6;
//             rec_count ++;
//             h ++;
//         }
//     }

//     fprintf(fp, "\n");

//     double frame_rate = 0.0;
//     double cycle_time = 0.0;

//     for(i = 0; i < num_exp * num_thread - startIdx-endIdx; i++)
//     {
//         if (new_sum_measure_data[i][19] > ACCEPTABLE_JITTER) continue;

//         if (i == 0) cycle_time = NAN;
//         else cycle_time = new_sum_measure_data[i][1] - new_sum_measure_data[i-1][1];

//         if (i == 0) frame_rate = NAN;
//         else frame_rate = 1000/cycle_time;

//         new_sum_measure_data[i][15] = cycle_time;
//         new_sum_measure_data[i][16] = frame_rate;
//         new_sum_measure_data[i][17] = (double)num_thread;
//         new_sum_measure_data[i][18] = R;

//         fprintf(fp, "%0.0f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,",  
//                 new_sum_measure_data[i][0], new_sum_measure_data[i][1], new_sum_measure_data[i][2], new_sum_measure_data[i][3], 
//                 new_sum_measure_data[i][4], new_sum_measure_data[i][5], new_sum_measure_data[i][6], new_sum_measure_data[i][7], 
//                 new_sum_measure_data[i][8], new_sum_measure_data[i][9], new_sum_measure_data[i][10], new_sum_measure_data[i][11], 
//                 new_sum_measure_data[i][12], new_sum_measure_data[i][13], new_sum_measure_data[i][14], new_sum_measure_data[i][15],
//                 new_sum_measure_data[i][16], new_sum_measure_data[i][17], new_sum_measure_data[i][18], new_sum_measure_data[i][19]);
    
//         for (int k = 0; k < data_count; ++k) {
//             fprintf(fp, "%.2f,", new_sum_measure_data[i][20 + k]);
//         }
//         fprintf(fp, "\n");
//     }

//     fclose(fp);

//     return 1;
// }

// void cpu_inference(network_state *state, network *net, layer *l, int split_index, int count, int thread_id)
// {
//     start_cpu_infer[split_index][count] = get_time_in_ms();
//     for(int j = infer_start[split_index]; j < infer_end[split_index]; j++) {
//         state->index = j;
//         l = &(net->layers[j]);
//         l->do_reclaiming = 0;
//         if(l->delta && state->train && l->train){
//             scal_cpu(l->outputs * l->batch, 0, l->delta, 1);
//         }
//         double layer_start = get_time_in_ms();
//         l->forward(*l, *state);
//         layer_time[j][count] = get_time_in_ms() - layer_start;

//         if(!isTest && layer_time[j][count] < max_layer_time[j]) {
//             while(layer_time[j][count] < max_layer_time[j]) {
//                 layer_time[j][count] = get_time_in_ms() - layer_start;
//             }
//         }

//         state->input = l->output;
//         if(j == net->n - 1) {
//             predictions = get_network_output(*net, 0);
//         }
//     }
//     end_cpu_infer[split_index][count] = get_time_in_ms();
// }

// void gpu_inference(network_state *state, network *net, layer *l, int split_index, int count, int thread_id)
// {
//     start_gpu_waiting[split_index][count] = get_time_in_ms();

//     pthread_mutex_lock(&mutex_gpu);

//     while(current_thread != thread_id) {
//         pthread_cond_wait(&cond, &mutex_gpu);
//     }
    
//     start_gpu_infer[split_index][count] = get_time_in_ms();

//     for(int j = infer_start[split_index]; j < infer_end[split_index]; j++) {
//         state->index = j;
//         l = &(net->layers[j]);
//         if(l->delta_gpu && state->train){
//             fill_ongpu(l->outputs * l->batch, 0, l->delta_gpu, 1);
//         }

//         double layer_start = get_time_in_ms();
//         l->forward_gpu(*l, *state);
//         CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
//         layer_time[j][count] = get_time_in_ms() - layer_start;

//         if(!isTest && (layer_time[j][count] < max_layer_time[j])) {
//             while(layer_time[j][count] < max_layer_time[j]) {
//                 layer_time[j][count] = get_time_in_ms() - layer_start;
//             }
//         }

//         state->input = l->output_gpu;
//         if(j == net->n - 1) {
//             predictions = get_network_output_gpu(*net);
//         }
//     }

//     current_thread = (current_thread) % num_thread + 1;
//     pthread_cond_broadcast(&cond);
//     end_gpu_infer[split_index][count] = get_time_in_ms();
//     pthread_mutex_unlock(&mutex_gpu);
// }

// void reclaiming_inference(network_state *state, network *net, layer *l, int split_index, int count, int thread_id)
// {
//     start_reclaim_waiting[split_index][count] = get_time_in_ms();

//     pthread_mutex_lock(&mutex_reclaim);

//     while(current_thread2 != thread_id) {
//         pthread_cond_wait(&cond2, &mutex_reclaim);
//     }

//     start_reclaim_infer[split_index][count] = get_time_in_ms();

//     for(int j = infer_start[split_index]; j < infer_end[split_index]; j++) {
//         state->index = j;
//         l = &(net->layers[j]);
//         l->do_reclaiming = 1;
//         if(l->delta && state->train && l->train){
//             scal_cpu(l->outputs * l->batch, 0, l->delta, 1);
//         }
//         double layer_start = get_time_in_ms();
//         l->forward(*l, *state);
//         layer_time[j][count] = get_time_in_ms() - layer_start;

//         if(!isTest && layer_time[j][count] < max_layer_time[j]) {
//             while(layer_time[j][count] < max_layer_time[j]) {
//                 layer_time[j][count] = get_time_in_ms() - layer_start;
//             }
//         }

//         state->input = l->output;
//         if(j == net->n - 1) {
//             predictions = get_network_output(*net, 0);
//         }
//     }

//     current_thread2 = (current_thread2) % num_thread + 1;
//     pthread_cond_broadcast(&cond2);
//     end_reclaim_infer[split_index][count] = get_time_in_ms();
//     pthread_mutex_unlock(&mutex_reclaim);
// }

// void initThread(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
//     float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
// {
//     g_datacfg = datacfg;
//     g_cfgfile = cfgfile;
//     g_weightfile = weightfile;
//     g_filename = filename;
//     g_thresh = thresh;
//     g_hier_thresh = hier_thresh;
//     g_dont_show = dont_show;
//     g_ext_output = ext_output;
//     g_save_labels = save_labels;
//     g_outfile = outfile;
//     g_letter_box = letter_box;
//     g_benchmark_layers = benchmark_layers;

//     R = 0.0;
//     max_preprocess_time = 0.0;
//     max_gpu_infer_time = 0.0;
//     max_reclaim_infer_time = 0.0;
//     max_cpu_infer_time = 0.0;
//     release_interval = 0.0;

//     options = read_data_cfg(g_datacfg);
//     name_list = option_find_str(options, "names", "data/names.list");
//     names_size = 0;
//     names = get_labels_custom(name_list, &names_size); //get_labels(name_list)
//     input = buff;

//     alphabet = load_alphabet();
//     object_detection = strstr(g_cfgfile, target_model);

//     if (g_filename) strncpy(input, g_filename, 256);
//     else printf("Error! File is not exist.");

// }

// void SetTest(bool test, bool set, bool reclaiming)
// {
//     isTest = test;
//     isSet = set;
//     isReclaiming = reclaiming;
    
//     nboxes = 0;
//     g_index = 0;

// }

// #ifdef GPU
// static void threadFunc(int arg)
// {
//     int thread_id = arg;

//     // __CPU AFFINITY SETTING__
//     cpu_set_t cpuset;
//     CPU_ZERO(&cpuset);
//     CPU_SET(coreIDOrder[thread_id], &cpuset); // cpu core index
//     int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
//     if (ret != 0) {
//         fprintf(stderr, "pthread_setaffinity_np() failed \n");
//         exit(0);
//     } 

//     // __GPU SETUP__
// #ifdef GPU   // GPU
//     if(gpu_index >= 0){
//         cuda_set_device(gpu_index);
//         CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
//     }
// #ifdef CUDNN_HALF
//     printf(" CUDNN_HALF=1 \n");
// #endif  // CUDNN_HALF
// #else
//     gpu_index = -1;
//     printf(" GPU isn't used \n");
//     init_cpu();
// #endif  // GPU

//     network net = net_list[thread_id];
//     layer l;

//     srand(2222222);
//     double remaining_time = 0.0;

//     for (int i = 0; i < num_exp; i++) {
//         int count = i * num_thread + thread_id - 1;

//         // // __Time Sync__
//         if(!isTest) {
//             if(i < START_INDEX) {
// 		        pthread_barrier_wait(&barrier);
//             }
//         }
//         else{
//             pthread_barrier_wait(&barrier);
//         }

//         // __Preprocess__

//         start_preprocess[count]=get_time_in_ms();

//         im = load_image(g_filename, 0, 0, net.c);
//         resized = resize_min(im, net.w);
//         cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
//         X = cropped.data;

//         end_preprocess[count] = get_time_in_ms();
//         e_preprocess[count] = end_preprocess[count] - start_preprocess[count];
//         e_preprocess_max[count] = get_time_in_ms() - start_preprocess[count];

//         start_infer[count] = get_time_in_ms();

//         // GPU Inference

//         if (net.gpu_index != cuda_get_device())
//             cuda_set_device(net.gpu_index);
//         int size = get_network_input_size(net) * net.batch;
//         network_state state;
//         state.index = 0;
//         state.net = net;
//         state.truth = 0;
//         state.train = 0;
//         state.delta = 0;
//         if(strcmp(inference_order[0], "GPU\0") == 0 && infer_end[0] != 0) {
//             state.input = net.input_state_gpu;
//             memcpy(net.input_pinned_cpu, X, size * sizeof(float));
//             cuda_push_array(state.input, net.input_pinned_cpu, size);
//         }
//         else state.input = X;

//         state.workspace = net.workspace;

//         for(int j = 0; j < NUM_SPLIT; ++j) {
//             if(strcmp(inference_order[j], "GPU\0") == 0) {
//                 gpu_inference(&state, &net, &l, j, count, thread_id);
//             }
//             else if(strcmp(inference_order[j], "CPU\0") == 0) {
//                 cpu_inference(&state, &net, &l, j, count, thread_id);
//             }
//             else if(strcmp(inference_order[j], "Reclaiming\0") == 0) {
//                 reclaiming_inference(&state, &net, &l, j, count, thread_id);
//             }
//             else {
//                 break;
//             } 
//         }

//         end_infer[count] = get_time_in_ms();

//         start_postprocess[count] = get_time_in_ms();

//         if(!object_detection) {
//             dets = get_network_boxes(&net, im.w, im.h, g_thresh,g_hier_thresh, 0, 1, &nboxes, g_letter_box);
//             if (nms) {
//                 if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
//                 else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
//             }
//             draw_detections_v3(im, dets, nboxes, g_thresh, names, alphabet, l.classes, g_ext_output);
//         }
//         else {
//             if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
//             top_k(predictions, net.outputs, top, indexes);
//             for(int j = 0; j < top; ++j){
//                 g_index = indexes[j];
//                 if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",g_index, names[g_index], predictions[g_index], (net.hierarchy->parent[g_index] >= 0) ? names[net.hierarchy->parent[g_index]] : "Root");
//                 else if (show_accuracy && thread_id == 1 && i == 3)printf("%s: %f\n",names[g_index], predictions[g_index]);
//             }
//         }



//         end_postprocess[count] = get_time_in_ms();
//         e_postprocess[count] = end_postprocess[count] - start_postprocess[count];
//         execution_time[count] = end_postprocess[count] - start_preprocess[count];

//         // __Measure Result__

//         core_id_list[count] = (double)sched_getcpu();
        
//         for(int j = 0; j < NUM_SPLIT; j++) {
//             waiting_gpu[j][count] = start_gpu_infer[j][count] - start_gpu_waiting[j][count];
//             e_gpu_infer[j][count] = end_gpu_infer[j][count] - start_gpu_infer[j][count];
//             waiting_reclaim[j][count] = start_reclaim_infer[j][count] - start_reclaim_waiting[j][count];//start_reclaim_waiting[i];
//             e_reclaim_infer[j][count] = end_reclaim_infer[j][count] - start_reclaim_infer[j][count];
//             e_cpu_infer[j][count] = end_cpu_infer[j][count] - start_cpu_infer[j][count];
//         }
//         e_infer[count] = end_infer[count] - start_infer[count];
//         execution_time_max[count] = get_time_in_ms() - start_preprocess[count];
//     }

//     pthread_exit(NULL);
// }


// void cpu_reclaiming(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
//     float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
// {
//     initThread(datacfg, cfgfile, weightfile, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers);
    
//     int visible_exp = show_result;
    
//     if (visible_exp) printf("\nCPU-Reclaiming with %d threads with %d gpu-layer & %d reclaim-layer\n", num_thread, gLayer, rLayer);

//     pthread_t threads[num_thread];
//     int rc;
//     int i;

//     if (opt_core > 0) num_thread = opt_core;

//     if (num_thread < (MAXCORES-1)/*&& (rLayer > 0)*/ ) {
    
//         R = 0.0;
//         max_preprocess_time = 0.0;
//         max_gpu_infer_time = 0.0;
//         max_reclaim_infer_time = 0.0;
//         max_cpu_infer_time = 0.0;
//         release_interval = 0.0;
    
//         // =====================RECLAMING=====================
//         if (visible_exp) printf("\n::TEST:: CPU-Reclaiming with %d threads with %d gpu-layer & %d reclaiming-layer\n", num_thread, gLayer, rLayer);
//         SetTest(true, false, true);
//         if(opt_core <= 0) isSet = true;

//         reset_check_jitter();
//         pthread_barrier_init(&barrier, NULL, num_thread);

//         for (i = 0; i < num_thread; i++) {
//             int threads_id = i + 1;

//             network net_init = parse_network_cfg_custom(g_cfgfile, 1, 1, device); // set batch=1

//             if (g_weightfile) {
//                 load_weights(&net_init, g_weightfile);
//             }

//             if (net_init.letter_box) g_letter_box = 1;
//             net_init.benchmark_layers = g_benchmark_layers;
//             fuse_conv_batchnorm(net_init);
//             calculate_binary_weights(net_init);

//             net_list[threads_id] = net_init;

//             if(i == 0) {
//                 num_network = net_init.n;  
//                 infer_end[0] = gLayer;
//                 infer_start[1] = gLayer;
//                 infer_end[1] = rLayer;
//                 infer_start[2] = rLayer;
//                 infer_end[2] = net_init.n;
//                 strcpy(inference_order[0], "GPU");
//                 strcpy(inference_order[1], "Reclaiming");
//                 strcpy(inference_order[2], "CPU");         
//                 strcpy(inference_order[3], "NULL");
//             }
            
//             cpu_set_t cpuset;

//             openblas_thread = (MAXCORES - 1) - num_thread + 1;
//             openblas_set_num_threads(openblas_thread);
//             for (int k = 0; k < openblas_thread - 1; k++) {
//                 CPU_ZERO(&cpuset);
//                 CPU_SET(coreIDOrder[(MAXCORES - 1) - k], &cpuset);
//                 openblas_setaffinity(k, sizeof(cpuset), &cpuset);
//             }

//             rc = pthread_create(&threads[i], NULL, threadFunc, threads_id);
//             if (rc) {
//                 printf("Error: Unable to create thread, %d\n", rc);
//                 exit(-1);
//             }
//         }

//         for (i = 0; i < num_thread; i++) {
//             pthread_join(threads[i], NULL);
//         }

//         for(int h = 0; h < num_network; h++) {	
//             for(int k = num_thread * START_INDEX; k < num_thread * (num_exp - END_INDEX); k++) {
//                 max_layer_time[h] += layer_time[h][k];
//                 division_count += 1;
//             }
//             max_layer_time[h] /= (float)division_count;
//             max_layer_time[h] *= 1.03;
//             division_count = 0;
//         }

//         max_preprocess_time = average(e_preprocess);
//         max_reclaim_infer_time = average(e_reclaim_infer);
//         max_gpu_infer_time = average(e_gpu_infer);
//         max_cpu_infer_time = average(e_cpu_infer);
//         execution_time_wo_waiting = max_gpu_infer_time + max_reclaim_infer_time + max_cpu_infer_time; // Delete Preprocess
//         R = maxOfThree(max_gpu_infer_time, max_reclaim_infer_time, execution_time_wo_waiting/num_thread);
//         release_interval = R * num_thread;
        
//         if (visible_exp) {
//             printf("e_pre : %0.02f, e_infer_cpu : %0.02f, e_infer_gpu : %0.02f, e_infer_reclaim : %0.02f, execution_time : %0.02f, TOTAL/N: %0.02f, Release interval: %0.02f\n", max_preprocess_time, max_cpu_infer_time, max_gpu_infer_time, max_reclaim_infer_time, execution_time_wo_waiting, execution_time_wo_waiting/num_thread, release_interval);
//         }

//         if (visible_exp) printf("\n::EXP-1:: CPU-Reclaiming with %d threads with %d gpu-layer & %d reclaiming-layer [R : %.2f]\n", num_thread, gLayer, rLayer, R);
//         pthread_barrier_init(&barrier, NULL, num_thread);
//         SetTest(false, true, true);
//         reset_check_jitter();
//         for (i = 0; i < num_thread; i++) {
//             int threads_id = i + 1;
//             rc = pthread_create(&threads[i], NULL, threadFunc, threads_id);
//             if (rc) {
//                 printf("Error: Unable to create thread, %d\n", rc);
//                 exit(-1);
//             }
//         }

//         for (i = 0; i < num_thread; i++) {
//             pthread_join(threads[i], NULL);
//         }

//         max_preprocess_time = average(e_preprocess);
//         max_reclaim_infer_time = average(e_reclaim_infer);
//         max_gpu_infer_time = average(e_gpu_infer);
//         max_cpu_infer_time = average(e_cpu_infer);
//         execution_time_wo_waiting = max_gpu_infer_time + max_reclaim_infer_time + max_cpu_infer_time; // Delete Preprocess
//         R = maxOfThree(max_gpu_infer_time, max_reclaim_infer_time, execution_time_wo_waiting/num_thread);
//         release_interval = R * num_thread;
        
//         if (visible_exp) {
//             printf("e_pre : %0.02f, e_infer_cpu : %0.02f, e_infer_gpu : %0.02f, e_infer_reclaim : %0.02f, execution_time : %0.02f, TOTAL/N: %0.02f, Release interval: %0.02f\n", max_preprocess_time, max_cpu_infer_time, max_gpu_infer_time, max_reclaim_infer_time, execution_time_wo_waiting, execution_time_wo_waiting/num_thread, release_interval);
//         }

//         if (visible_exp) printf("\n::EXP-2:: CPU-Reclaiming with %d threads with %d gpu-layer & %d reclaiming-layer [R : %.2f]\n", num_thread, gLayer, rLayer, R);
//         pthread_barrier_init(&barrier, NULL, num_thread);
//         SetTest(false, true, true);
//         reset_check_jitter();
//         for (i = 0; i < num_thread; i++) {
//             int threads_id = i + 1;
//             rc = pthread_create(&threads[i], NULL, threadFunc, threads_id);
//             if (rc) {
//                 printf("Error: Unable to create thread, %d\n", rc);
//                 exit(-1);
//             }
//         }

//         for (i = 0; i < num_thread; i++) {
//             pthread_join(threads[i], NULL);
//         }

//         max_preprocess_time = average(e_preprocess);
//         max_reclaim_infer_time = average(e_reclaim_infer);
//         max_gpu_infer_time = average(e_gpu_infer);
//         max_cpu_infer_time = average(e_cpu_infer);
//         execution_time_wo_waiting = max_gpu_infer_time + max_reclaim_infer_time + max_cpu_infer_time; // Delete Preprocess
//         R = maxOfThree(max_gpu_infer_time, max_reclaim_infer_time, execution_time_wo_waiting/num_thread);
//         release_interval = R * num_thread;
        
//         if (visible_exp) {
//             printf("e_pre : %0.02f, e_infer_cpu : %0.02f, e_infer_gpu : %0.02f, e_infer_reclaim : %0.02f, execution_time : %0.02f, TOTAL/N: %0.02f, Release interval: %0.02f\n", max_preprocess_time, max_cpu_infer_time, max_gpu_infer_time, max_reclaim_infer_time, execution_time_wo_waiting, execution_time_wo_waiting/num_thread, release_interval);
//         }
//         if(write_result_reclaiming() == -1) {
//             /* return error */
//             exit(0);
//         }

//         if(write_acceleration_info() == -1) {
//             exit(0);
//         }
//     }

//     pthread_barrier_destroy(&barrier);
    
//     return 0;

// }
// #else

// void cpu_reclaiming(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
//     float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
// {
//     printf("!!ERROR!! GPU = 0 \n");
// }
// #endif  // GPU
