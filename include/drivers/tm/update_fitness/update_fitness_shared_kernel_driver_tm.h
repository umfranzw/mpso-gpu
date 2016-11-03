#ifndef _UPDATE_FITNESS_KERNEL_DRIVER_TM_H_
#define _UPDATE_FITNESS_KERNEL_DRIVER_TM_H_

#include "CL/cl.h"
#include "devices.h"
#include "config_tm.h"
#include "kernels_tm.h"
#include "buffers_tm.h"

void set_update_fitness_shared_kernel_args_tm(
    config_tm *conf,
    cl_kernel *kernel,
    mpso_bufs_tm *bufs,
    cl_uint swarms_per_group
    );

void launch_update_fitness_shared_kernel_tm(
    config_tm *conf,
    cl_kernel *kernel,
    mpso_bufs_tm *bufs,
    cl_uint iter_num,
    cl_uint swapped_last_iter,
    device *dev,
    char *kernel_label,
    cl_uint combined
    );

void launch_update_fitness_shared_vec_kernel_tm(
    config_tm *conf,
    cl_kernel *kernels,
    mpso_bufs_tm *bufs,
    cl_uint iter_num,
    cl_uint swapped_last_iter,
    device *dev,
    cl_uint combined
    );

/* void launch_update_fitness_shared_unvec_kernel_tm( */
/*     config_tm *conf, */
/*     cl_kernel *kernels, */
/*     mpso_bufs_tm *bufs, */
/*     mpso_events_tm *events, */
/*     cl_uint iter_num, */
/*     cl_uint swapped_last_iter, */
/*     device *dev, */
/*     cl_uint combined */
/*     ); */

#endif
