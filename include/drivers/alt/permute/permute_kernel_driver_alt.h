#ifndef _PERMUTE_KERNEL_DRIVER_ALT_H_
#define _PERMUTE_KERNEL_DRIVER_ALT_H_

#define _CRT_RAND_S //for rand_s() windows crypto library fuction (must appear before #include statements)

#include "CL/cl.h"
#include "global_constants.h"
#include "config_alt.h"
#include "kernels_alt.h"
#include "buffers_alt.h"
#include "devices.h"

void set_permute_kernel_args_alt(
    config_alt *conf,
    cl_kernel *kernel,
    mpso_bufs_alt *bufs,
    cl_uint iter_index
    );

void launch_permute_kernel_alt(
    config_alt *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_alt *bufs,
    cl_uint iter_index,
    device *dev
    );
    
#endif
