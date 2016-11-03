#ifndef _F7_KERNEL_DRIVER_COMMON_H_
#define _F7_KERNEL_DRIVER_COMMON_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include ALG_HEADER_STR_COMMON(config)
#include ALG_HEADER_STR_COMMON(kernels)
#include ALG_HEADER_STR_COMMON(buffers)
#include "drivers/wg_sizer.h"

void set_f7_kernel_args(
    ALG_NAME_COMMON(config) *conf,
    cl_kernel *kernel,
    ALG_NAME_COMMON(mpso_bufs) *bufs,
    cl_uint particles_per_group,
    cl_uint threads_per_particle,
    cl_uint scratch_per_particle
    );

void launch_f7_kernel(
    ALG_NAME_COMMON(config) *conf,
    cl_kernel *kernels,
    ALG_NAME_COMMON(mpso_bufs) *bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
