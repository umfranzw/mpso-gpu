#ifndef _BUFFERS_MCS_H_
#define _BUFFERS_MCS_H_

#include "CL/cl.h"
#include "CL/cl_ext.h"
#include "config_mcs.h"
#include "devices.h"
#include "bench_fcns.h"
#include "utils.h"
#include "buffer_utils.h"

typedef struct mpso_bufs_mcs
{
    cl_mem positions_buf;
    cl_mem velocities_buf;
    cl_mem fitnesses_buf;
    cl_mem pbest_positions_buf;
    cl_mem sbest_positions_buf;
    cl_mem pbest_fitnesses_buf;
    cl_mem sbest_fitnesses_buf;
    cl_mem fitness_scratch_buf;
    cl_mem best_indices_buf;
    cl_mem worst_indices_buf;
    cl_mem optimum_buf;
    cl_mem initial_rot_matrix_buf;
    cl_mem rot_matrix_buf;
    cl_mem perm_vec_buf;
    cl_mem crossover_perm_buf;
    cl_mem swarm_health_buf;
    cl_mem pre_mut_fit_buf;
    cl_mem pre_mut_pos_buf;
    cl_mem pre_mut_vel_buf;
    cl_mem fitness_sample_buf;
    cl_mem final_fitness_buf;
    cl_mem global_scratch_buf;
    cl_mem extra_data_buf;
    //cl_mem mut_counts_buf;
    
    //cl_mem test_buf;
} mpso_bufs_mcs;

void create_mpso_bufs_mcs(
    config_mcs *conf,
    bench_fcn_info *bench_info,
    mpso_bufs_mcs *bufs,
    profiling_data *prof,
    device *gpu
    );

void release_mpso_bufs_mcs(
    config_mcs *conf,
    bench_fcn_info *bench_info,
    mpso_bufs_mcs *bufs
    );

/* void fill_crossover_perm_buf( */
/*     mpso_bufs_mcs *bufs, */
/*     config_mcs *conf, */
/*     device *gpu */
/*     ); */

#endif
