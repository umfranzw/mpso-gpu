#ifndef _CROSS_MUT_TOURN_KERNEL_DRIVER_GA_H_
#define _CROSS_MUT_TOURN_KERNEL_DRIVER_GA_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "config_ga.h"
#include "kernels_ga.h"
#include "buffers_ga.h"
#include "devices.h"
#include "drivers/wg_sizer.h"

void set_cross_mut_tourn_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint threads_per_particle,
    cl_uint local_mem_per_group,
    cl_uint iter_index
    );

void launch_cross_mut_tourn_kernel_ga(
    config_ga *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_ga *bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
