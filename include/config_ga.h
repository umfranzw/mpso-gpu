#ifndef _CONFIG_GA_H_
#define _CONFIG_GA_H_

#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"
#include "global_constants.h"
#include "config_utils.h"

typedef struct config_ga
{
    cl_uint num_swarms;
    cl_uint num_sparticles;
    cl_uint num_dims;
    cl_uint max_iters;
    cl_uint max_ga_init_iters;
    cl_float omega;
    cl_float omega_decay;
    cl_float c1;
    cl_float c2;
    cl_uint bench_fcn;
    cl_uint m;
    cl_float max_axis_val;
    cl_uint ga_cross_ratio;
    cl_uint ga_mut_prob;
    cl_uint ga_tourn_size;
    cl_uint exchange_iters;
    cl_uint num_exchange;
    cl_uint cross_iters;
    cl_float unhealthy_ratio;
    cl_uint unhealthy_iters;
    cl_float mut_prob;
    cl_uint num_reps;
    cl_uint seed;
    cl_float max_vel;
    cl_uint fitness_sample_interval;
} config_ga;

cl_int parse_config_ga(
    FILE *file, 
    config_ga *conf
    );

cl_int get_next_config_ga(
    FILE *config_file,
    config_ga *config
    );

#endif
