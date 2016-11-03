#ifndef _MUT_RESTORE_KERNEL_DRIVER_GA_H_
#define _MUT_RESTORE_KERNEL_DRIVER_GA_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include "config_ga.h"
#include "kernels_ga.h"
#include "buffers_ga.h"
#include "drivers/wg_sizer.h"

void set_mut_restore_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint particles_per_group
    );

void launch_mut_restore_kernel_ga(
    config_ga *conf,
    cl_kernel *kernels,
    mpso_bufs_ga *bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
