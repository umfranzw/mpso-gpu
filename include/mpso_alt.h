#ifndef _MPSO_ALT_H_
#define _MPSO_ALT_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "bench_fcns.h"
#include "buffers_alt.h"
#include "config_alt.h"
#include "kernels_alt.h"
#include "timer.h"
#include "pretty_printer.h"

#include "drivers/common/particle_init/particle_init_kernel_driver_common.h"
#include "drivers/common/update_fitness/update_fitness_shared_kernel_driver_common.h"
#include "drivers/common/update_best_vals/update_best_vals_kernel_driver_common.h"
#include "drivers/common/find_best_worst/find_best_worst_kernel_driver_common.h"
#include "drivers/common/swap_particles/swap_particles_kernel_driver_common.h"
#include "drivers/common/init_rot_matrix/init_rot_matrix_kernel_driver_common.h"
#include "drivers/common/find_min/find_min_kernel_driver_common.h"
#include "drivers/common/update_samples/update_samples_kernel_driver_common.h"

#include "drivers/alt/permute/permute_kernel_driver_alt.h"
#include "drivers/alt/cross_mut/cross_mut_pbest_kernel_driver_alt.h"
#include "drivers/alt/cross_mut/cross_mut_tourn_kernel_driver_alt.h"
#include "drivers/alt/update_pos_vel/update_pos_vel_kernel_driver_alt.h"
#include "drivers/common/update_pos_vel/update_pos_vel_kernel_driver_common.h"

void get_swarm_config_str(
    config_alt *conf,
    cl_mem *buf,
    char *swarm_config_str,
    device *dev
    );

void run_mpso_alt(
    config_alt *conf,
    cl_uint config_index,
    cl_uint num_configs,
    cl_program *program,
    cl_kernel *kernel_buf,
    device *cpu,
    device *gpu
    );

#endif
