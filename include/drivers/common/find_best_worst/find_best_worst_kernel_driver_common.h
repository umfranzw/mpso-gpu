#ifndef _FIND_BEST_WORST_KERNEL_DRIVER_COMMON_H_
#define _FIND_BEST_WORST_KERNEL_DRIVER_COMMON_H_

#include "CL/cl.h"
#include "global_constants.h"
#include ALG_HEADER_STR(config)
#include ALG_HEADER_STR(kernels)
#include ALG_HEADER_STR(buffers)
#include "utils.h"
#include "devices.h"
#include "drivers/wg_sizer.h"

void launch_find_best_worst_kernel_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
    device *dev,
    size_t *global_work_size,
    size_t *local_work_size,
    char *kernel_label
    );

void set_find_best_worst_alt_kernel_args_common(
    ALG_NAME(config) *generic_conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *generic_bufs,
    cl_uint swarms_per_group
    );

void launch_find_best_worst_alt_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_num,
    device *dev
    );

void launch_find_best_worst_alt2_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_num,
    device *dev
    );

void set_find_best_worst_vec2_kernel_args_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
    cl_uint swarms_per_group
    );

void launch_find_best_worst_vec2_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_num,
    device *dev
    );

#endif
