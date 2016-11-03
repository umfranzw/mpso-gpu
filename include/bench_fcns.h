#ifndef _BENCH_FCNS_H_
#define _BENCH_FCNS_H_

#include "CL/cl.h"
#include "global_constants.h"
#include ALG_HEADER_STR(kernels)

//warning: if this doesn't exactly match the number of initialized array elements in init_bench_fcn_info, it will corrupt your stack in bizzare ways!
#define NUM_BENCH_FCNS 24

typedef struct bench_fcn_info
{
    cl_float max_axis_val;
    cl_uint need_rot_matrix;
    cl_uint need_perm_vec;
    cl_uint need_opt_vec;
} bench_fcn_info;

void init_bench_fcn_info(
    bench_fcn_info *info
    );

#endif
