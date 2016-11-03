#ifndef _PERMUTE_KERNEL_DRIVER_MCS_H_
#define _PERMUTE_KERNEL_DRIVER_MCS_H_

#define _CRT_RAND_S //for rand_s() windows crypto library fuction (must appear before #include statements)

#include "CL/cl.h"
#include "global_constants.h"
#include "config_mcs.h"
#include "kernels_mcs.h"
#include "buffers_mcs.h"
#include "devices.h"

void set_permute_kernel_args_mcs(
    config_mcs *conf,
    cl_kernel *kernel,
    mpso_bufs_mcs *bufs,
    cl_uint iter_index
    );

void launch_permute_kernel_mcs(
    config_mcs *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_mcs *bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
