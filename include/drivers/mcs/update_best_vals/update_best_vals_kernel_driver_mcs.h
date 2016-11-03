#ifndef _UPDATE_BEST_VALS_KERNEL_DRIVER_MCS_H_
#define _UPDATE_BEST_VALS_KERNEL_DRIVER_MCS_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include "config_mcs.h"
#include "kernels_mcs.h"
#include "buffers_mcs.h"
#include "drivers/wg_sizer.h"

void set_update_best_vals_vec_kernel_args_mcs(
    config_mcs *conf,
    cl_kernel *kernel,
    mpso_bufs_mcs *bufs,
    cl_uint swarms_per_group
    );

void calc_update_best_vals_sizes_mcs(
    config_mcs *conf,
    size_t *local_work_size,
    size_t *global_work_size,
    cl_uint *swarms_per_group,
    device *dev
    );

void launch_update_best_vals_vec_kernel_mcs(
    config_mcs *conf,
    cl_kernel *kernels,
    mpso_bufs_mcs *bufs,
    cl_uint iter_num,
    device *dev
    );

#endif
