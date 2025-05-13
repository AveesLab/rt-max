#ifndef DETECTOR_H
#define DETECTOR_H

#include "image.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef MEASURE
#define MAX(x,y) (((x) < (y) ? (y) : (x)))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAXCORES 8
#define MEASUREMENT_PATH "measure"
#endif

void pipeline(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void pipeline_jitter(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void sequential(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void data_parallel(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void gpu_accel(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int theoretical_exp, int theo_thread, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

extern int isGPU;
extern int r_time;
extern int num_exp;
extern int core_id;
extern int num_blas;
extern int num_thread;
extern int num_process;
extern int gLayer;
extern int rLayer;
extern int Gstart;
extern int Gend;
extern int skip_layers[1000][10];

#ifdef __cplusplus
}
#endif

#endif
