#ifndef _PARTICLE_INIT_KERNEL_DRIVER_MCS_H_
#define _PARTICLE_INIT_KERNEL_DRIVER_MCS_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include "config_mcs.h"
#include "kernels_mcs.h"
#include "buffers_mcs.h"

void set_particle_init_kernel_args_mcs(
    config_mcs *conf,
    cl_kernel *kernel,
    mpso_bufs_mcs *bufs,
    cl_uint rep
    );

void launch_particle_init_vec_kernel_mcs(
    config_mcs *conf,
    cl_kernel *kernels,
    mpso_bufs_mcs *bufs,
    cl_uint rep,
    device *dev
    );

#endif
