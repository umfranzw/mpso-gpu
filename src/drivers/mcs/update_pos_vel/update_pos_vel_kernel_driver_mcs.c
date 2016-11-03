#include "drivers/mcs/update_pos_vel/update_pos_vel_kernel_driver_mcs.h"

void set_update_pos_vel_vec_kernel_args_mcs(
    config_mcs *conf,
    cl_kernel *kernel,
    mpso_bufs_mcs *bufs,
    cl_uint iter_index,
    cl_uint rep
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
        &(bufs->swarm_health_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->fitnesses_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pre_mut_pos_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pre_mut_vel_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pre_mut_fit_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->extra_data_buf)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
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
        sizeof(cl_float),
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
        sizeof(cl_uint),
        &(conf->unhealthy_iters)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &(conf->mut_prob)
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
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &rep
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d", arg_index);
}

void launch_update_pos_vel_vec_kernel_mcs(
    config_mcs *conf,
    cl_kernel *kernels,
    mpso_bufs_mcs *bufs,
    cl_uint iter_index,
    cl_uint rep,
    device *dev
    )
{
    size_t global_work_size = (size_t) (conf->num_swarms * conf->num_sparticles * conf->num_dims / 4);
    
    #if LAUNCH_WARNINGS
    printf("Launching update_pos_vel_mcs kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: NULL\n");
    #endif
    
    set_update_pos_vel_vec_kernel_args_mcs(
        conf,
        &(kernels[UPDATE_POS_VEL_VEC_KERNEL_MCS]),
        bufs,
        iter_index,
        rep
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[UPDATE_POS_VEL_VEC_KERNEL_MCS],
        1,
        NULL,
        &global_work_size,
        (size_t *) NULL,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching update_pos_vel_mcs kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
