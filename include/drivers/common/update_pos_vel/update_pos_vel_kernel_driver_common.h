#ifndef _UPDATE_POS_VEL_KERNEL_DRIVER_COMMON_H_
#define _UPDATE_POS_VEL_KERNEL_DRIVER_COMMON_H_

#include "CL/cl.h"
#include "global_constants.h"
#include ALG_HEADER_STR(config)
#include ALG_HEADER_STR(buffers)
#include "devices.h"
#include ALG_HEADER_STR(kernels)

void set_update_pos_vel_vec_kernel_args_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
    cl_uint iter_index
    );

void launch_update_pos_vel_kernel_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
    cl_uint iter_index,
    device *dev,
    cl_uint *global_work_size,
    cl_uint *local_work_size,
    char *kernel_label
    );

void launch_update_pos_vel_vec_kernel_common(
   void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
