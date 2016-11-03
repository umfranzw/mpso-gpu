#include "drivers/common/update_pos_vel/update_pos_vel_kernel_driver_common.h"

void set_update_pos_vel_vec_kernel_args_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
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
        sizeof(cl_float),
        &(conf->max_vel)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d", arg_index);
}

void launch_update_pos_vel_kernel_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
    cl_uint iter_index,
    device *dev,
    cl_uint *global_work_size,
    cl_uint *local_work_size,
    char *kernel_label
    )
{
    #if LAUNCH_WARNINGS
    printf("Launching %s kernel.\n", kernel_label);
    printf("global_work_size: %u\n", global_work_size == NULL ? 0 : *global_work_size);
    printf("local_work_size: %u\n", local_work_size == NULL ? 0 : *local_work_size);
    #endif
    
    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        *kernel,
        1,
        NULL,
        global_work_size,
        local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching %s kernel.", kernel_label);

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}

void launch_update_pos_vel_vec_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_index,
    device *dev
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;
    ALG_NAME(mpso_bufs) *bufs = (ALG_NAME(mpso_bufs) *) generic_bufs;

    size_t global_work_size = (size_t) (conf->num_swarms * conf->num_sparticles * conf->num_dims / 4);

    set_update_pos_vel_vec_kernel_args_common(
        conf,
        &(kernels[ALG_NAME_CAPS(UPDATE_POS_VEL_VEC_KERNEL)]),
        bufs,
        iter_index
        );

    launch_update_pos_vel_kernel_common(
        conf,
        &(kernels[ALG_NAME_CAPS(UPDATE_POS_VEL_VEC_KERNEL)]),
        bufs,
        iter_index,
        dev,
        &global_work_size,
        (size_t *) NULL,
        "update_pos_vel_vec"
        );
}
