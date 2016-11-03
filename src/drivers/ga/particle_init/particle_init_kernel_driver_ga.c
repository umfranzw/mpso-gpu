#include "drivers/ga/particle_init/particle_init_kernel_driver_ga.h"

void set_particle_init_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint rep
    )
{
    cl_uint arg_index = 0;
    cl_int error;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->positions_buf)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->velocities_buf)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pbest_positions_buf)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pbest_fitnesses_buf)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->sbest_positions_buf)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->sbest_fitnesses_buf)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->extra_data_buf)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->swarm_health_buf)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(rep)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &(conf->max_axis_val)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->seed)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->max_vel)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_reps)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->alg_health_buf)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
}

void launch_particle_init_vec_kernel_ga(
    config_ga *conf,
    cl_kernel *kernels,
    mpso_bufs_ga *bufs,
    device *dev,
    cl_uint rep
    )
{
    size_t global_worksize = (size_t) (conf->num_swarms * conf->num_sparticles * conf->num_dims / 4);

    #if LAUNCH_WARNINGS
    printf("Launching particle_init_vec kernel.\n");
    printf("global_work_size: %u\n", global_worksize);
    printf("local_work_size: NULL\n");
    #endif
    
    set_particle_init_kernel_args_ga(
        conf,
        &(kernels[PARTICLE_INIT_VEC_KERNEL_GA]),
        bufs,
        rep
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[PARTICLE_INIT_VEC_KERNEL_GA],
        1,
        NULL,
        &global_worksize,
        NULL,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching particle_init_vec kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
