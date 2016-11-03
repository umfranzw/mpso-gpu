#include "buffers_tm.h"

void create_mpso_bufs_tm(
    config_tm *conf,
    mpso_bufs_tm *bufs,
    profiling_data *prof,
    device *gpu
    )
{
    cl_int error;

    //create buffers
    bufs->positions_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating positions buffer.");

    bufs->velocities_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating velocities buffer.");

    bufs->fitnesses_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * conf->num_sparticles * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating fitnesses buffer.");

    bufs->pbest_positions_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating pbest positions buffer.");

    bufs->sbest_positions_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * conf->num_dims * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating sbest positions buffer.");

    bufs->pbest_fitnesses_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * conf->num_sparticles * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating pbest fitnesses buffer.");

    bufs->sbest_fitnesses_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * sizeof(cl_float),
        NULL,
        &error
        );
    check_error(error, "Error creating sbest fitnesses buffer.");

    bufs->best_indices_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * conf->num_exchange * sizeof(cl_uint),
        NULL,
        &error
        );
    check_error(error, "Error creating best_indices_buf.");

    bufs->worst_indices_buf = clCreateBuffer(gpu->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
        conf->num_swarms * conf->num_exchange * sizeof(cl_uint),
        NULL,
        &error
        );
    check_error(error, "Error creating worst_indices_buf.");

    bufs->etc_buf = clCreateBuffer(gpu->context,
                                   CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD,
                                   conf->num_dims * conf->num_machines * sizeof(cl_float),
                                   NULL,
                                   &error
        );
    check_error(error, "Error creating etc buffer.");

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

    /* bufs->test_buf = clCreateBuffer(gpu->context, */
    /*                                 CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD, */
    /*                                 (conf->num_swarms * conf->num_sparticles * conf->num_dims) * sizeof(cl_float), */
    /*                                 NULL, */
    /*                                 &error */
    /*     ); */
    /* check_error(error, "Error creating test buffer."); */
}

void release_mpso_bufs_tm(
    mpso_bufs_tm *bufs
    )
{
    cl_int error = clReleaseMemObject(bufs->positions_buf);
    check_error(error, "Error releasing positions_buf.");
    
    error = clReleaseMemObject(bufs->velocities_buf);
    check_error(error, "Error releasing velocities_buf.");
    
    error = clReleaseMemObject(bufs->fitnesses_buf);
    check_error(error, "Error releasing fitnesses_buf.");
    
    error = clReleaseMemObject(bufs->pbest_positions_buf);
    check_error(error, "Error releasing pbest_positions_buf.");
    
    error = clReleaseMemObject(bufs->sbest_positions_buf);
    check_error(error, "Error releasing sbest_positions_buf.");
    
    error = clReleaseMemObject(bufs->pbest_fitnesses_buf);
    check_error(error, "Error releasing pbest_fitnesses_buf.");
    
    error = clReleaseMemObject(bufs->sbest_fitnesses_buf);
    check_error(error, "Error releasing sbest_fitnesses_buf.");
    
    error = clReleaseMemObject(bufs->etc_buf);
    check_error(error, "Error releasing etc_buf.");
    
    error = clReleaseMemObject(bufs->best_indices_buf);
    check_error(error, "Error releasing best_indices_buf.");
    
    error = clReleaseMemObject(bufs->worst_indices_buf);
    check_error(error, "Error releasing worst_indices_buf.");

    error = clReleaseMemObject(bufs->fitness_sample_buf);
    check_error(error, "Error releasing buffer.");

    error = clReleaseMemObject(bufs->final_fitness_buf);
    check_error(error, "Error releasing buffer.");

    error = clReleaseMemObject(bufs->global_scratch_buf);
    check_error(error, "Error releasing buffer.");
    
    /* error = clReleaseMemObject(bufs->test_buf); */
    /* check_error("Error releasing buffer."); */
}
