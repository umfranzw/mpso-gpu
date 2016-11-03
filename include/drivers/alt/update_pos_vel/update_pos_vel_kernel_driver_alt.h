#ifndef _UPDATE_POS_VEL_KERNEL_DRIVER_ALT_H_
#define _UPDATE_POS_VEL_KERNEL_DRIVER_ALT_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "config_alt.h"
#include "kernels_alt.h"
#include "buffers_alt.h"
#include "devices.h"

void set_update_pos_vel_vec_kernel_args_alt(
    config_alt *conf,
    cl_kernel *kernel,
    mpso_bufs_alt *bufs,
    cl_uint iter_index
    );

void launch_update_pos_vel_vec_kernel_alt(
    config_alt *conf,
    cl_kernel *kernels,
    mpso_bufs_alt *bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
