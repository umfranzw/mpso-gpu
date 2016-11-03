#ifndef _SWAP_PARTICLES_KERNEL_DRIVER_COMMON_H_
#define _SWAP_PARTICLES_KERNEL_DRIVER_COMMON_H_

#include "CL/cl.h"
#include "global_constants.h"
#include ALG_HEADER_STR(config)
#include ALG_HEADER_STR(kernels)
#include ALG_HEADER_STR(buffers)
#include "devices.h"
#include "drivers/wg_sizer.h"

void set_swap_particles_vec_kernel_args_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
    cl_uint swarms_per_group
    );

void launch_swap_particles_vec_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_num,
    device *dev
    );

#endif
