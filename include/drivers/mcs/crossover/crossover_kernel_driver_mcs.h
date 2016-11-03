#ifndef _CROSSOVER_KERNEL_DRIVER_MCS_H_
#define _CROSSOVER_KERNEL_DRIVER_MCS_H_

#include "CL/cl.h"
#include "devices.h"
#include "config_mcs.h"
#include "kernels_mcs.h"
#include "buffers_mcs.h"

void set_crossover_kernel_args_mcs(
    config_mcs *conf,
    cl_kernel *kernel,
    mpso_bufs_mcs *bufs
    );

void launch_crossover_kernel_mcs(
    config_mcs *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_mcs *bufs,
    device *dev
    );

#endif
