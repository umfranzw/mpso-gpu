#ifndef _INIT_ROT_MATRIX_KERNEL_DRIVER_COMMON_H_
#define _INIT_ROT_MATRIX_KERNEL_DRIVER_COMMON_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include ALG_HEADER_STR_COMMON(config)
#include ALG_HEADER_STR_COMMON(kernels)
#include ALG_HEADER_STR_COMMON(buffers)

void set_init_rot_matrix_args_common(
    ALG_NAME_COMMON(config) *conf,
    cl_kernel *kernel,
    ALG_NAME_COMMON(mpso_bufs) *bufs,
    cl_uint width,
    cl_uint height,
    cl_uint block_size
    );

void launch_init_rot_matrix_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    device *dev
    );

#endif
