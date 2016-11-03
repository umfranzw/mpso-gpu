#ifndef _BUFFERS_TM_H_
#define _BUFFERS_TM_H_

#include "CL/cl.h"
#include "CL/cl_ext.h"
#include "config_tm.h"
#include "devices.h"
#include "bench_fcns.h"

typedef struct mpso_bufs_tm
{
    cl_mem positions_buf;
    cl_mem velocities_buf;
    cl_mem fitnesses_buf;
    cl_mem pbest_positions_buf;
    cl_mem sbest_positions_buf;
    cl_mem pbest_fitnesses_buf;
    cl_mem sbest_fitnesses_buf;
    cl_mem etc_buf;
    cl_mem best_indices_buf;
    cl_mem worst_indices_buf;
    cl_mem num_swarms_buf;
    cl_mem num_sparticles_buf;
    cl_mem fitness_sample_buf;
    cl_mem final_fitness_buf;
    cl_mem global_scratch_buf;
    
    //cl_mem test_buf;
} mpso_bufs_tm;

void create_mpso_bufs_tm(
    config_tm *conf,
    mpso_bufs_tm *bufs,
    profiling_data *prof,
    device *gpu
    );

void release_mpso_bufs_tm(
    mpso_bufs_tm *bufs
    );

#endif
