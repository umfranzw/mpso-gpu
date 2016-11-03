#ifndef _PARTICLE_INIT_KERNEL_DRIVER_GA_H_
#define _PARTICLE_INIT_KERNEL_DRIVER_GA_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include "config_ga.h"
#include "kernels_ga.h"
#include "buffers_ga.h"

void set_particle_init_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint rep
    );

void launch_particle_init_vec_kernel_ga(
    config_ga *conf,
    cl_kernel *kernels,
    mpso_bufs_ga *bufs,
    device *dev,
    cl_uint rep
    );

#endif
