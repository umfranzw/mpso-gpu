#ifndef _CROSS_MUT_PBEST_KERNEL_DRIVER_GA_H_
#define _CROSS_MUT_PBEST_KERNEL_DRIVER_GA_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "config_ga.h"
#include "kernels_ga.h"
#include "buffers_ga.h"
#include "devices.h"

void set_cross_mut_pbest_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint iter_index
    );

void launch_cross_mut_pbest_kernel_ga(
    config_ga *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_ga *bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
