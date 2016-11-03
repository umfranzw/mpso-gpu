#ifndef _FIND_MIN_KERNEL_DRIVER_COMMON_H_
#define _FIND_MIN_KERNEL_DRIVER_COMMON_H_

#include "CL/cl.h"
#include "buffer_utils.h"
#include "devices.h"
#include ALG_HEADER_STR_COMMON(kernels)

void set_find_min_kernel_driver_common_args(
    cl_kernel *kernel,
    cl_mem *input_buf,
    cl_uint input_len,
    cl_mem *global_scratch_buf,
    cl_mem *result_buf,
    cl_uint result_index,
    cl_uint local_mem_elements
    );

void find_min_cpu(
    cl_mem *input_buf,
    cl_uint input_len,
    cl_mem *results_buf,
    cl_uint result_index,
    device *dev
    );

void do_kernel_launch(
    cl_kernel *kernels,
    cl_mem *input_buf,
    cl_uint input_len,
    cl_mem *global_scratch_buf,
    cl_mem *result_buf,
    cl_uint result_index,
    device *dev
    );

void launch_find_min_vec_kernel_common(
    cl_kernel *kernels,
    cl_mem *input_buf,
    cl_uint input_len,
    cl_mem *global_scratch_buf,
    cl_mem *result_buf,
    cl_uint result_index,
    device *dev
    );

#endif
