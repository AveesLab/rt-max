#ifndef DETECTOR_H
#define DETECTOR_H

#include "image.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef MEASURE
#define MAX(x,y) (((x) < (y) ? (y) : (x)))
#define MAXCORES 12
#define MEASUREMENT_PATH "measure"
#endif

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void sequential(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void sequential_multiblas(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void data_parallel(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void data_parallel_mp(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void gpu_accel(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int theoretical_exp, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void gpu_accel_mp(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void cpu_reclaiming(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

void cpu_reclaiming_mp(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);

extern int num_exp;
extern int core_id;
extern int num_blas;
extern int num_thread;
extern int num_process;
extern int gLayer;
extern int rLayer;
extern int skip_layers[1000];

#ifdef __cplusplus
}
#endif

#endif
