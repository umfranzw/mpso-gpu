#ifndef _CONFIG_TM_H_
#define _CONFIG_TM_H_

#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"
#include "global_constants.h"
#include "config_utils.h"

typedef struct config_tm
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
    cl_uint num_machines;
    cl_uint num_reps;
    cl_float max_axis_val;
    cl_uint seed;
    cl_uint fitness_sample_interval;
} config_tm;

cl_int get_next_config_tm(
    FILE *config_file,
    config_tm *config
    );

cl_int parse_config_tm(
    FILE *file, 
    config_tm *conf
    );

#endif
