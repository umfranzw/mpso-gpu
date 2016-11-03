#ifndef _UPDATE_POS_VEL_HYBRID_KERNEL_DRIVER_TM_H_
#define _UPDATE_POS_VEL_HYBRID_KERNEL_DRIVER_TM_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "config_tm.h"
#include "buffers_tm.h"
#include "devices.h"
#include "kernels_tm.h"

void set_update_pos_vel_hybrid_kernel_args_tm(
    config_tm *conf,
    cl_kernel *kernel,
    mpso_bufs_tm *bufs,
    cl_uint iter_index,
    cl_uint group_size
    );

void launch_update_pos_vel_hybrid_kernel_tm(
    config_tm conf,
    cl_kernel *kernels,
    mpso_bufs_tm bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
