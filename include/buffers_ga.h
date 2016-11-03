#ifndef _BUFFERS_GA_H_
#define _BUFFERS_GA_H_

#include "CL/cl.h"
#include "CL/cl_ext.h"
#include "config_ga.h"
#include "devices.h"
#include "bench_fcns.h"
#include "utils.h"
#include "buffer_utils.h"

typedef struct mpso_bufs_ga
{
    cl_mem positions_buf;
    cl_mem velocities_buf;
    cl_mem fitnesses_buf;
    cl_mem pbest_positions_buf;
    cl_mem sbest_positions_buf;
    cl_mem pbest_fitnesses_buf;
    cl_mem sbest_fitnesses_buf;
    cl_mem best_indices_buf;
    cl_mem worst_indices_buf;
    cl_mem optimum_buf;
    cl_mem initial_rot_matrix_buf;
    cl_mem rot_matrix_buf;
    cl_mem perm_vec_buf;
    cl_mem fitness_sample_buf;
    cl_mem final_fitness_buf;
    cl_mem extra_data_buf;
    cl_mem global_scratch_buf;
    cl_mem crossover_perm_buf;
    cl_mem swarm_health_buf;
    cl_mem pre_mut_fit_buf;
    cl_mem pre_mut_pos_buf;
    cl_mem pre_mut_vel_buf;
    cl_mem mut_counts_buf;
    cl_mem alg_health_buf;

    //cl_mem test_buf;
    
    /* cl_mem test_tourn_indices_buf; */
    /* cl_mem test_tourn_vals_buf; */
    /* cl_mem test_tourn_sel_buf; */
    /* cl_mem test_tourn_mins_buf; */
    /* cl_mem test_local_mem_buf; */
} mpso_bufs_ga;

void create_mpso_bufs_ga(
    config_ga *conf,
    bench_fcn_info *bench_info,
    mpso_bufs_ga *bufs,
    profiling_data *prof,
    device *gpu
    );

void release_mpso_bufs_ga(
    config_ga *conf,
    bench_fcn_info *bench_info,
    mpso_bufs_ga *bufs
    );

#endif
