#ifndef _UPDATE_POS_VEL_KERNEL_DRIVER_MCS_H_
#define _UPDATE_POS_VEL_KERNEL_DRIVER_MCS_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include "config_mcs.h"
#include "kernels_mcs.h"
#include "buffers_mcs.h"

void set_update_pos_vel_vec_kernel_args_mcs(
    config_mcs *conf,
    cl_kernel *kernel,
    mpso_bufs_mcs *bufs,
    cl_uint iter_index,
    cl_uint rep
    );

void launch_update_pos_vel_vec_kernel_mcs(
    config_mcs *conf,
    cl_kernel *kernels,
    mpso_bufs_mcs *bufs,
    cl_uint iter_index,
    cl_uint rep,
    device *dev
    );

#endif
