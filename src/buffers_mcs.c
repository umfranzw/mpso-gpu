#include "buffers_mcs.h"

void create_mpso_bufs_mcs(
    config_mcs *conf,
    bench_fcn_info *bench_info,
    mpso_bufs_mcs *bufs,
    profiling_data *prof,
    device *gpu
    )
{
    #if LAUNCH_WARNINGS
    printf("Creating buffers.\n");
    #endif

    cl_int error;

    //create buffers
    bufs->positions_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,// | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating positions buffer.");

    bufs->velocities_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating velocities buffer.");

    bufs->fitnesses_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,// | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_sparticles * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating fitnesses buffer.");

    bufs->pbest_positions_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating pbest positions buffer.");

    bufs->sbest_positions_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_dims * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating sbest positions buffer.");

    bufs->pbest_fitnesses_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_sparticles * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating pbest fitnesses buffer.");

    bufs->sbest_fitnesses_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating sbest fitnesses buffer.");

    bufs->best_indices_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_exchange * sizeof(cl_uint),
        NULL,
        &error
        );
    check_error(error, "Error creating best_indices_buf.");

    bufs->worst_indices_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_exchange * sizeof(cl_uint),
        NULL,
        &error
        );
    check_error(error, "Error creating worst_indices_buf.");

    if (bench_info[conf->bench_fcn - 1].need_opt_vec)
    {
        bufs->optimum_buf = clCreateBuffer(
            gpu->context,
            CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD,
            conf->num_dims * sizeof(cl_float),
            NULL,
            &error
            );
        check_error(error, "Error creating optimum_buf.");
    }
    
    if (bench_info[conf->bench_fcn - 1].need_rot_matrix)
    {
        bufs->initial_rot_matrix_buf = clCreateBuffer(
            gpu->context,
            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
            conf->m * conf->m * sizeof(cl_float),
            NULL,
            &error
            );
        check_error(error, "Error creating initial_rot_matrix_buf.");

        bufs->rot_matrix_buf = clCreateBuffer(
            gpu->context,
            CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD | CL_MEM_HOST_NO_ACCESS,
            conf->m * conf->m * sizeof(cl_float),
            NULL,
            &error
            );
        check_error(error, "Error creating rot_matrix_buf.");
    }

    if (bench_info[conf->bench_fcn - 1].need_perm_vec)
    {
        bufs->perm_vec_buf = clCreateBuffer(
            gpu->context,
            CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD,
            conf->num_dims * sizeof(cl_uint),
            NULL,
            &error
            );
        check_error(error, "Error creating perm_vec_buf.");
    }

    bufs->crossover_perm_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * conf->num_sparticles * sizeof(cl_uint),
        NULL,
        &error
        );
    check_error(error, "Error creating crossover_perm_buf.");

    bufs->swarm_health_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * sizeof(cl_uint),
        NULL,
        &error
        );
    check_error(error, "Error creating swarm_health_buf.");

    bufs->pre_mut_fit_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,// | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_sparticles * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating pre_mut_fit_buf.");

    bufs->pre_mut_pos_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD, // | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating pre_mut_pos_buf.");

    bufs->pre_mut_vel_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD | CL_MEM_HOST_NO_ACCESS,
        conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating pre_mut_vel_buf.");

    bufs->fitness_sample_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        sizeof(cl_float) * prof->num_fitness_samples * conf->num_reps,
        NULL,
        &error
        );
    check_error(error, "Error creating fitness_sample buffer.");

    bufs->final_fitness_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        sizeof(cl_float) * conf->num_reps,
        NULL,
        &error
        );
    check_error(error, "Error creating final_fitness buffer.");

    
    cl_uint global_scratch_buf_size;
    if (conf->num_swarms * conf->num_sparticles / 4 < GPU_WORKGROUP_SIZE)
    {
        global_scratch_buf_size = 1; //one workgroup
    }
    else
    {
        global_scratch_buf_size = (conf->num_swarms * conf->num_sparticles / 4) / GPU_WORKGROUP_SIZE + ((conf->num_swarms * conf->num_sparticles / 4) % GPU_WORKGROUP_SIZE ? 1 : 0); //multiple workgroups, plus potentially one extra that may not be filled
    }

    bufs->global_scratch_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        sizeof(cl_float) * global_scratch_buf_size,
        NULL,
        &error
        );
    check_error(error, "Error creating global_scratch buffer.");

    bufs->extra_data_buf = clCreateBuffer(
        gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        prof->extra_data_elements * conf->num_reps * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating extra_data buffer.");
    
    /* bufs->mut_counts_buf = clCreateBuffer(gpu->context, */
    /*     CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD, */
    /*     conf->num_reps * conf->num_swarms * sizeof(cl_float), */
    /*     NULL, */
    /*     &error */
    /*     ); */
    /* check_error(error, "Error creating mut_counts buffer."); */
    
    /* bufs->test_buf = clCreateBuffer(gpu->context, */
    /*     CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD, */
    /*     conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float), */
    /*     NULL, */
    /*     &error */
    /*     ); */
    /* check_error(error, "Error creating test buffer."); */

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}

void release_mpso_bufs_mcs(
    config_mcs *conf,
    bench_fcn_info *bench_info,
    mpso_bufs_mcs *bufs
    )
{
    cl_int error = clReleaseMemObject(bufs->positions_buf);
    check_error(error, "Error releasing buffer.");
    
    error = clReleaseMemObject(bufs->velocities_buf);
    check_error(error, "Error releasing buffer.");
    
    error = clReleaseMemObject(bufs->fitnesses_buf);
    check_error(error, "Error releasing buffer.");
    
    error = clReleaseMemObject(bufs->pbest_positions_buf);
    check_error(error, "Error releasing buffer.");
    
    error = clReleaseMemObject(bufs->sbest_positions_buf);
    check_error(error, "Error releasing buffer.");
    
    error = clReleaseMemObject(bufs->pbest_fitnesses_buf);
    check_error(error, "Error releasing buffer.");
    
    error = clReleaseMemObject(bufs->sbest_fitnesses_buf);
    check_error(error, "Error releasing buffer.");
    
    /* error = clReleaseMemObject(bufs->rands_buf); */
    /* check_error(error, "Error releasing buffer."); */

    if (bench_info[conf->bench_fcn - 1].need_opt_vec)
    {
        error = clReleaseMemObject(bufs->optimum_buf);
        check_error(error, "Error releasing buffer.");
    }
    
    error = clReleaseMemObject(bufs->best_indices_buf);
    check_error(error, "Error releasing buffer.");
    
    error = clReleaseMemObject(bufs->worst_indices_buf);
    check_error(error, "Error releasing buffer.");

    if (bench_info[conf->bench_fcn - 1].need_rot_matrix)
    {
        error = clReleaseMemObject(bufs->initial_rot_matrix_buf);
        check_error(error, "Error releasing buffer.");
        
        error = clReleaseMemObject(bufs->rot_matrix_buf);
        check_error(error, "Error releasing buffer.");
    }

    if (bench_info[conf->bench_fcn - 1].need_perm_vec)
    {
        error = clReleaseMemObject(bufs->perm_vec_buf);
        check_error(error, "Error releasing buffer.");
    }

    error = clReleaseMemObject(bufs->crossover_perm_buf);
    check_error(error, "Error releasing buffer.");

    error = clReleaseMemObject(bufs->swarm_health_buf);
    check_error(error, "Error releasing swarm_health buffer.");

    error = clReleaseMemObject(bufs->pre_mut_fit_buf);
    check_error(error, "Error releasing pre_mut_fit buffer.");

    error = clReleaseMemObject(bufs->pre_mut_pos_buf);
    check_error(error, "Error releasing pre_mut_pos buffer.");

    error = clReleaseMemObject(bufs->pre_mut_vel_buf);
    check_error(error, "Error releasing pre_mut_vel buffer.");

    error = clReleaseMemObject(bufs->fitness_sample_buf);
    check_error(error, "Error releasing buffer.");

    error = clReleaseMemObject(bufs->final_fitness_buf);
    check_error(error, "Error releasing buffer.");

    error = clReleaseMemObject(bufs->global_scratch_buf);
    check_error(error, "Error releasing buffer.");

    error = clReleaseMemObject(bufs->extra_data_buf);
    check_error(error, "Error releasing buffer.");

    /* error = clReleaseMemObject(bufs->test_buf); */
    /* check_error(error, "Error releasing buffer."); */
}

/* void fill_crossover_perm_buf( */
/*     mpso_bufs_mcs *bufs, */
/*     config_mcs *conf, */
/*     device *gpu */
/*     ) */
/* { */
/*     cl_event event; */
/*     cl_uint *host_buf = (cl_uint *) map_buffer( */
/*         &(bufs->crossover_perm_buf), */
/*         CL_MAP_WRITE_INVALIDATE_REGION, */
/*         conf->num_swarms * conf->num_sparticles * sizeof(cl_uint), */
/*         gpu, */
/*         &event */
/*         ); */
/*     clWaitForEvents(1, &event); */

/*     cl_uint i; */
/*     for (i = 0; i < conf->num_swarms; i++) */
/*     { */
/*         fill_perm_buf( */
/*             host_buf + i * conf->num_sparticles, */
/*             conf->num_sparticles */
/*             ); */
/*     } */

/*     unmap_buffer( */
/*         &(bufs->crossover_perm_buf), */
/*         host_buf, */
/*         gpu, */
/*         &event */
/*         ); */
/*     clWaitForEvents(1, &event); */

/*     clReleaseEvent(event); */
/* } */

