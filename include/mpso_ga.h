#ifndef _MPSO_GA_H_
#define _MPSO_GA_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "bench_fcns.h"
#include "buffers_ga.h"
#include "kernels_ga.h"
#include "config_ga.h"
#include "timer.h"
#include "pretty_printer.h"

#include "drivers/common/init_rot_matrix/init_rot_matrix_kernel_driver_common.h"
#include "drivers/common/update_fitness/update_fitness_shared_kernel_driver_common.h"
#include "drivers/common/find_best_worst/find_best_worst_kernel_driver_common.h"
#include "drivers/common/find_min/find_min_kernel_driver_common.h"
#include "drivers/common/update_samples/update_samples_kernel_driver_common.h"
#include "drivers/common/swap_particles/swap_particles_kernel_driver_common.h"

#include "drivers/ga/cross_mut/cross_mut_tourn_kernel_driver_ga.h"
#include "drivers/ga/permute/permute_kernel_driver_ga.h"
#include "drivers/ga/crossover/crossover_kernel_driver_ga.h"
#include "drivers/ga/mut_restore/mut_restore_kernel_driver_ga.h"
#include "drivers/ga/particle_init/particle_init_kernel_driver_ga.h"
#include "drivers/ga/update_pos_vel/update_pos_vel_kernel_driver_ga.h"

//this is for the GA
//#include "drivers/common/update_best_vals/update_best_vals_kernel_driver_common.h"
#include "drivers/ga/update_best_vals/update_best_vals_kernel_driver_spacial.h"

//this is for MPSO-MCS
#include "drivers/ga/update_best_vals/update_best_vals_kernel_driver_ga.h"


#define GA_FITNESS_SAMPLE_PTS 1
#define ALG_INTERVAL 1000

void run_mpso_ga(
    config_ga *conf,
    cl_uint config_index,
    cl_uint num_configs,
    cl_program *program,
    cl_kernel *kernel_buf,
    device *cpu,
    device *gpu
    );

void do_ga_alg(
    config_ga *conf,
    profiling_data *prof_data,
    cl_kernel *kernel_buf,
    mpso_bufs_ga *bufs,
    cl_uint rep,
    device *sel_dev,
    cl_uint rel_i,
    cl_uint abs_i
    //cl_uint recalc_sizing
    );

void do_mpso_alg(
    config_ga *conf,
    profiling_data *prof_data,
    cl_kernel *kernel_buf,
    mpso_bufs_ga *bufs,
    cl_uint rep,
    device *gpu,
    cl_uint rel_i,
    cl_uint abs_i
    //cl_uint recalc_sizing,
    //cl_uint last_shift_iter
    );

#endif
