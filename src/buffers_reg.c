#include "buffers_reg.h"

void create_mpso_bufs_reg(
    config_reg *conf,
    bench_fcn_info *bench_info,
    mpso_bufs_reg *bufs,
    profiling_data *prof,
    device *gpu
    )
{
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

    bufs->worst_indices_buf = clCreateBuffer(
        gpu->context,
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
            CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD | CL_MEM_HOST_NO_ACCESS,
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

    /* bufs->test_buf = clCreateBuffer( */
    /*     gpu->context, */
    /*     CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD, */
    /*     conf->num_swarms * conf->num_sparticles / 2 * sizeof(cl_float), */
    /*     NULL, */
    /*     &error */
    /*     ); */
    /* check_error(error, "Error creating test buffer."); */
}

void release_mpso_bufs_reg(
    config_reg *conf,
    bench_fcn_info *bench_info,
    mpso_bufs_reg *bufs
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

    error = clReleaseMemObject(bufs->fitness_sample_buf);
    check_error(error, "Error releasing buffer.");

    error = clReleaseMemObject(bufs->final_fitness_buf);
    check_error(error, "Error releasing buffer.");

    error = clReleaseMemObject(bufs->global_scratch_buf);
    check_error(error, "Error releasing buffer.");

    /* error = clReleaseMemObject(bufs->test_buf); */
    /* check_error(error, "Error releasing buffer."); */
}
