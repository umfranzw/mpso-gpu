#include "drivers/alt/update_pos_vel/update_pos_vel_kernel_driver_alt.h"

void set_update_pos_vel_vec_kernel_args_alt(
    config_alt *conf,
    cl_kernel *kernel,
    mpso_bufs_alt *bufs,
    cl_uint iter_index
    )
{
    cl_int error;
    cl_uint arg_index = 0;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->positions_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->velocities_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pbest_positions_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->sbest_positions_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->swarm_types_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->omega)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &(conf->c1)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &(conf->c2)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &(conf->max_axis_val)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->seed)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &iter_index
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &(conf->max_vel)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d", arg_index);    
}

void launch_update_pos_vel_vec_kernel_alt(
    config_alt *conf,
    cl_kernel *kernels,
    mpso_bufs_alt *bufs,
    cl_uint iter_index,
    device *dev
    )
{
    size_t global_work_size = (size_t) (conf->num_swarms * conf->num_sparticles * conf->num_dims / 4);
    
    set_update_pos_vel_vec_kernel_args_alt(
        conf,
        &(kernels[UPDATE_POS_VEL_VEC_KERNEL_ALT]),
        bufs,
        iter_index
        );

    #if LAUNCH_WARNINGS
    char *name = get_kernel_name(&(kernels[UPDATE_POS_VEL_VEC_KERNEL_ALT]));
    printf("Launching %s kernel.\n", name);
    free(name);
    printf("global_work_size: %u\n", global_work_size == NULL ? 0 : global_work_size);
    printf("local_work_size: 0\n");
    #endif
    
    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[UPDATE_POS_VEL_VEC_KERNEL_ALT],
        1,
        NULL,
        &global_work_size,
        NULL,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching update_pos_vel_vec kernel.");
 
    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
