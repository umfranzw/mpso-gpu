#include "drivers/alt/permute/permute_kernel_driver_alt.h"

void set_permute_kernel_args_alt(
    config_alt *conf,
    cl_kernel *kernel,
    mpso_bufs_alt *bufs,
    cl_uint iter_index
    )
{
    cl_uint arg_index = 0;
    cl_int error;
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->crossover_perm_buf)
        );
    check_error(error, "Error setting permute kernel arg %d", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        conf->num_sparticles * sizeof(cl_uint),
        NULL
        );
    check_error(error, "Error setting permute kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        conf->num_sparticles * sizeof(cl_uint),
        NULL
        );
    check_error(error, "Error setting permute kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &iter_index //launched once every iteration
        );
    check_error(error, "Error setting permute kernel arg %d", arg_index);
    arg_index++;

    cl_uint seed;
    rand_s(&seed);
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &seed
        );
    check_error(error, "Error setting permute kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting permute kernel arg %d", arg_index);
}

void launch_permute_kernel_alt(
    config_alt *conf,
    cl_kernel *kernel_buf,
    mpso_bufs_alt *bufs,
    cl_uint iter_index,
    device *dev
    )
{
    size_t global_work_size = (size_t) conf->num_swarms * conf->num_sparticles / 2;
    size_t local_work_size = (size_t) conf->num_sparticles / 2;
    
    #if LAUNCH_WARNINGS
    printf("Launching permute kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: %u\n", local_work_size);
    #endif

    set_permute_kernel_args_alt(
        conf,
        &(kernel_buf[PERMUTE_VEC_KERNEL_ALT]),
        bufs,
        iter_index
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernel_buf[PERMUTE_VEC_KERNEL_ALT],
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching permute kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
