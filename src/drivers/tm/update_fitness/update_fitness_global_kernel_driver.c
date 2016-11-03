#include "drivers/tm/update_fitness/update_fitness_global_kernel_driver.h"

void set_update_fitness_global_vec_args(
    config *conf,
    cl_kernel *kernel,
    mpso_bufs *bufs,
    cl_uint group_size
    )
{
    cl_int error;
    cl_uint arg_index = 0;
    
    error = clSetKernelArg(*kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->positions_buf)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(*kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->fitnesses_buf)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(*kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->fitness_scratch_buf)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(*kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->config_buf)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(*kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->etc_buf)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
}

void launch_update_fitness_global_vec_kernel(
    config *conf,
    cl_kernel *kernel,
    mpso_bufs *bufs,
    device *dev
    )
{
    cl_uint swarms_per_group = dev->max_workgroup_size / conf->num_sparticles > conf->num_swarms ? conf->num_swarms : dev->max_workgroup_size / conf->num_sparticles;
    size_t local_work_size = swarms_per_group * conf->num_sparticles;
    size_t global_work_size = local_work_size * (conf->num_swarms / swarms_per_group);
    
    #ifdef LAUNCH_WARNINGS
    printf("Launching update_fitness_global_vec kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: %u\n", local_work_size);
    #endif

    set_update_fitness_global_vec_args(conf, kernel, bufs, (cl_uint) local_work_size);

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        *kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching update_fitness_global kernel.");
}

void launch_update_fitness_global_unvec_kernel(
    config *conf,
    cl_kernel *kernel,
    mpso_bufs *bufs,
    device *dev
    )
{
    #ifdef LAUNCH_WARNINGS
    printf("Launching update_fitness_global_unvec kernel.\n");
    #endif

    launch_update_fitness_global_vec_kernel(conf, kernel, bufs, dev);
}
