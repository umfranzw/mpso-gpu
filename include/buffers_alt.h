#ifndef _BUFFERS_ALT_H_
#define _BUFFERS_ALT_H_

#include "CL/cl.h"
#include "CL/cl_ext.h"
#include "config_alt.h"
#include "devices.h"
#include "bench_fcns.h"
#include "utils.h"
#include "buffer_utils.h"

typedef struct mpso_bufs_alt
{
    cl_mem positions_buf;
    cl_mem velocities_buf;
    cl_mem fitnesses_buf;
    cl_mem pbest_positions_buf;
    cl_mem sbest_positions_buf;
    cl_mem pbest_fitnesses_buf;
    cl_mem sbest_fitnesses_buf;
    cl_mem crossover_perm_buf;
    cl_mem fitness_scratch_buf;
    cl_mem best_indices_buf;
    cl_mem worst_indices_buf;
    cl_mem optimum_buf;
    cl_mem initial_rot_matrix_buf;
    cl_mem rot_matrix_buf;
    cl_mem perm_vec_buf;
    cl_mem swarm_types_buf;
    cl_mem fitness_sample_buf;
    cl_mem final_fitness_buf;
    cl_mem global_scratch_buf;
    
    //cl_mem test_buf;
} mpso_bufs_alt;

void create_mpso_bufs_alt(
    config_alt *conf,
    bench_fcn_info *bench_info,
    mpso_bufs_alt *bufs,
    profiling_data *prof,
    device *gpu
    );

void release_mpso_bufs_alt(
    config_alt *conf,
    bench_fcn_info *bench_info,
    mpso_bufs_alt *bufs
    );

void fill_swarm_types_buf(
    config_alt *conf,
    cl_mem *buf,
    device *gpu
    );

#endif
