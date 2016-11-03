#include "drivers/ga/cross_mut/cross_mut_pbest_kernel_driver_ga.h"

void set_cross_mut_pbest_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint iter_index
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
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pbest_positions_buf)
        );
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->crossover_perm_buf)
        );
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &(conf->max_axis_val)
        );
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &(conf->ga_cross_ratio)
        );
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &(conf->ga_mut_prob)
        );
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(iter_index)
        );
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->seed)
        );
    check_error(error, "Error setting cross_mut kernel arg %d", arg_index);
}

void launch_cross_mut_pbest_kernel_ga(
    config_ga *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_ga *bufs,
    cl_uint iter_index,
    device *dev
    )
{
    static size_t local_work_size;
    static size_t global_work_size;
    static size_t particles_per_group;
    static size_t threads_per_particle;
    static size_t local_mem_per_particle;

    if (iter_index == 0)
    {
        //figure out launch dimensions
        global_work_size = conf->num_swarms * conf->num_sparticles * conf->num_dims / 4;
    }
    
    #if LAUNCH_WARNINGS
    printf("Launching cross_mut_pbest kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: NULL\n");
    #endif

    set_cross_mut_pbest_kernel_args_ga(
        conf,
        &(kernel_buf[CROSS_MUT_VEC_KERNEL_GA]),
        bufs,
        iter_index
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernel_buf[CROSS_MUT_VEC_KERNEL_GA],
        1,
        NULL,
        &global_work_size,
        NULL,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching cross_mut kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
