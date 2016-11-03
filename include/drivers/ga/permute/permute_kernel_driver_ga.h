#ifndef _PERMUTE_KERNEL_DRIVER_GA_H_
#define _PERMUTE_KERNEL_DRIVER_GA_H_

#define _CRT_RAND_S //for rand_s() windows crypto library fuction (must appear before #include statements)

#include "CL/cl.h"
#include "global_constants.h"
#include "config_ga.h"
#include "kernels_ga.h"
#include "buffers_ga.h"
#include "devices.h"

void set_permute_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint iter_index
    );

void launch_permute_kernel_ga(
    config_ga *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_ga *bufs,
    cl_uint iter_index,
    device *dev
    );
    
#endif
