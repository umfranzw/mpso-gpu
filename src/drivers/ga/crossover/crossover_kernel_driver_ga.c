#include "drivers/ga/crossover/crossover_kernel_driver_ga.h"

void set_crossover_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs
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
    check_error(error, "Error setting crossover kernel arg %d", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pbest_positions_buf)
        );
    check_error(error, "Error setting crossover kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->crossover_perm_buf)
        );
    check_error(error, "Error setting crossover kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting crossover kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting crossover kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting crossover kernel arg %d", arg_index);
}

void launch_crossover_kernel_ga(
    config_ga *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_ga *bufs,
    device *dev
    )
{
    size_t global_work_size = (size_t) (conf->num_swarms * conf->num_sparticles * conf->num_dims / 4);
    
    #if LAUNCH_WARNINGS
    printf("Launching crossover kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: 0\n");
    #endif

    set_crossover_kernel_args_ga(
        conf,
        &(kernel_buf[CROSSOVER_VEC_KERNEL_GA]),
        bufs
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernel_buf[CROSSOVER_VEC_KERNEL_GA],
        1,
        NULL,
        &global_work_size,
        NULL,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching crossover kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
