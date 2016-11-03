#ifndef _MPSO_TM_H_
#define _MPSO_TM_H_

#include "pretty_printer.h"
#include "buffers_tm.h"
#include "config_tm.h"
#include "kernels_tm.h"
#include "buffer_utils.h"

#include "drivers/common/particle_init/particle_init_kernel_driver_common.h"
#include "drivers/common/update_best_vals/update_best_vals_kernel_driver_common.h"
#include "drivers/common/update_pos_vel/update_pos_vel_kernel_driver_common.h"
#include "drivers/common/find_best_worst/find_best_worst_kernel_driver_common.h"
#include "drivers/common/find_min/find_min_kernel_driver_common.h"
#include "drivers/common/update_samples/update_samples_kernel_driver_common.h"

#include "drivers/tm/swap_particles/swap_particles_kernel_driver_tm.h"
#include "drivers/tm/update_fitness/update_fitness_shared_kernel_driver_tm.h"

void run_mpso_tm(
    config_tm *conf,
    cl_uint config_index,
    cl_uint num_configs,
    cl_program *program,
    cl_kernel *kernel_buf,
    device *cpu,
    device *gpu
    );

void gen_etc_matrix(
    config_tm *conf,
    mpso_bufs_tm *bufs,
    device *gpu
    );

#endif
