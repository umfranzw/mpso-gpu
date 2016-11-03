#ifndef _CROSS_MUT_TOURN_KERNEL_DRIVER_ALT_H_
#define _CROSS_MUT_TOURN_KERNEL_DRIVER_ALT_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "config_alt.h"
#include "kernels_alt.h"
#include "buffers_alt.h"
#include "devices.h"
#include "drivers/wg_sizer.h"

void set_cross_mut_tourn_kernel_args_alt(
    config_alt *conf,
    cl_kernel *kernel,
    mpso_bufs_alt *bufs,
    cl_uint threads_per_particle,
    cl_uint local_mem_per_particle,
    cl_uint iter_index
    );

void launch_cross_mut_tourn_kernel_alt(
    config_alt *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_alt *bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
