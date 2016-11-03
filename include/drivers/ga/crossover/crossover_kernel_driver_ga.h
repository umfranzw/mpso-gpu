#ifndef _CROSSOVER_KERNEL_DRIVER_GA_H_
#define _CROSSOVER_KERNEL_DRIVER_GA_H_

#include "CL/cl.h"
#include "devices.h"
#include "config_ga.h"
#include "kernels_ga.h"
#include "buffers_ga.h"

void set_crossover_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs
    );

void launch_crossover_kernel_ga(
    config_ga *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_ga *bufs,
    device *dev
    );

#endif
