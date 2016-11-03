#ifndef _MPSO_MCS_H_
#define _MPSO_MCS_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "timer.h"
#include "bench_fcns.h"
#include "buffers_mcs.h"
#include "kernels_mcs.h"
#include "config_mcs.h"
#include "pretty_printer.h"

#include "drivers/common/update_fitness/update_fitness_shared_kernel_driver_common.h"
#include "drivers/mcs/update_pos_vel/update_pos_vel_kernel_driver_mcs.h"
#include "drivers/common/find_best_worst/find_best_worst_kernel_driver_common.h"
#include "drivers/common/swap_particles/swap_particles_kernel_driver_common.h"
#include "drivers/common/init_rot_matrix/init_rot_matrix_kernel_driver_common.h"
#include "drivers/common/find_min/find_min_kernel_driver_common.h"
#include "drivers/common/update_samples/update_samples_kernel_driver_common.h"

#include "drivers/mcs/permute/permute_kernel_driver_mcs.h"
#include "drivers/mcs/crossover/crossover_kernel_driver_mcs.h"
#include "drivers/mcs/mut_restore/mut_restore_kernel_driver_mcs.h"
#include "drivers/mcs/particle_init/particle_init_kernel_driver_mcs.h"
#include "drivers/mcs/update_best_vals/update_best_vals_kernel_driver_mcs.h"

void run_mpso_mcs(
    config_mcs *conf,
    cl_uint config_index,
    cl_uint num_configs,
    cl_program *program,
    cl_kernel *kernel_buf,
    device *cpu,
    device *gpu
    );

#endif
