#ifndef _CONFIG_MCS_H_
#define _CONFIG_MCS_H_

#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"
#include "global_constants.h"
#include "config_utils.h"

typedef struct config_mcs
{
    cl_uint num_swarms;
    cl_uint num_sparticles;
    cl_uint num_dims;
    cl_uint max_iters;
    cl_float omega;
    cl_float omega_decay;
    cl_float c1;
    cl_float c2;
    cl_uint exchange_iters;
    cl_uint num_exchange;
    cl_uint num_reps;
    cl_uint bench_fcn;
    cl_uint m;
    cl_float max_axis_val;
    cl_uint cross_iters;
    cl_float unhealthy_ratio;
    cl_uint unhealthy_iters;
    cl_float mut_prob;
    cl_uint seed;
    cl_float max_vel;
    cl_uint fitness_sample_interval;
} config_mcs;

cl_int parse_config_mcs(
    FILE *file, 
    config_mcs *conf
    );

cl_int get_next_config_mcs(
    FILE *config_file,
    config_mcs *config
    );

#endif
