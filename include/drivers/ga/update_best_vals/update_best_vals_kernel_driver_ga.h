#ifndef _UPDATE_BEST_VALS_KERNEL_DRIVER_GA_H_
#define _UPDATE_BEST_VALS_KERNEL_DRIVER_GA_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include "config_ga.h"
#include "kernels_ga.h"
#include "buffers_ga.h"
#include "drivers/wg_sizer.h"

void set_update_best_vals_vec_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint swarms_per_group
    );

void calc_update_best_vals_sizes_ga(
    config_ga *conf,
    size_t *local_work_size,
    size_t *global_work_size,
    cl_uint *swarms_per_group,
    device *dev
    );

void launch_update_best_vals_vec_kernel_ga(
    config_ga *conf,
    cl_kernel *kernels,
    mpso_bufs_ga *bufs,
    cl_uint iter_num,
    device *dev
    );

#endif
