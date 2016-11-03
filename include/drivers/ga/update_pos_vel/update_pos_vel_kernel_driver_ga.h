#ifndef _UPDATE_POS_VEL_KERNEL_DRIVER_GA_H_
#define _UPDATE_POS_VEL_KERNEL_DRIVER_GA_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include "config_ga.h"
#include "kernels_ga.h"
#include "buffers_ga.h"

void launch_update_pos_vel_vec_kernel_ga(
    config_ga *conf,
    cl_kernel *kernels,
    mpso_bufs_ga *bufs,
    cl_uint iter_index,
    cl_uint rep,
    device *dev
    );

void set_update_pos_vel_vec_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint iter_index,
    cl_uint rep
    );

#endif
