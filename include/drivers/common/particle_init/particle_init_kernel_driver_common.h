#ifndef _PARTICLE_INIT_KERNEL_DRIVER_COMMON_H_
#define _PARTICLE_INIT_KERNEL_DRIVER_COMMON_H_

#include "CL/cl.h"
#include "global_constants.h"
#include ALG_HEADER_STR(config)
#include ALG_HEADER_STR(kernels)
#include ALG_HEADER_STR(buffers)
#include "devices.h"

void set_particle_init_kernel_args_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs
    );

void launch_particle_init_vec_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    device *dev
    );

void launch_particle_init_kernel_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
    device *dev,
    cl_uint *global_worksize,
    cl_uint *local_worksize,
    char *kernel_label
    );

#endif
