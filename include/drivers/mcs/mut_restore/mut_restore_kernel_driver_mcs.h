#ifndef _MUT_RESTORE_KERNEL_DRIVER_MCS_H_
#define _MUT_RESTORE_KERNEL_DRIVER_MCS_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include "config_mcs.h"
#include "kernels_mcs.h"
#include "buffers_mcs.h"
#include "drivers/wg_sizer.h"

void set_mut_restore_kernel_args_mcs(
    config_mcs *conf,
    cl_kernel *kernel,
    mpso_bufs_mcs *bufs,
    cl_uint particles_per_group
    );

void launch_mut_restore_kernel_mcs(
    config_mcs *conf,
    cl_kernel *kernels,
    mpso_bufs_mcs *bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
