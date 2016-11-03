#ifndef _UPDATE_SAMPLES_KERNEL_DRIVER_COMMON_H_
#define _UPDATE_SAMPLES_KERNEL_DRIVER_COMMON_H_

#include "CL/cl.h"
#include "buffer_utils.h"
#include "devices.h"
#include ALG_HEADER_STR_COMMON(kernels)

void set_update_samples_vec_kernel_driver_common_args(
    cl_kernel *kernel,
    cl_mem *samples_buf,
    cl_mem *src_buf,
    cl_uint num_samples,
    cl_uint sample_index,
    cl_uint init_output,
    cl_float divisor
    );

void launch_update_samples_vec_kernel_common(
    cl_kernel *kernels,
    cl_mem *samples_buf,
    cl_mem *src_buf,
    cl_uint num_samples,
    cl_uint sample_index,
    cl_uint init_output,
    cl_float divisor,
    device *dev
    );

#endif
