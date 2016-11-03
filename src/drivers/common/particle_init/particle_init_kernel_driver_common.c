#include "drivers/common/particle_init/particle_init_kernel_driver_common.h"

void set_particle_init_kernel_args_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs
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
        sizeof(cl_uint),
        &(conf->num_dims)
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
        sizeof(cl_float),
        &(conf->max_vel)
        );
    check_error(error, "Error setting particle init kernel arg %d", arg_index);
}

void launch_particle_init_vec_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    device *dev
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;
    ALG_NAME(mpso_bufs) *bufs = (ALG_NAME(mpso_bufs) *) generic_bufs;

    size_t work_items_needed = (size_t) (conf->num_swarms * conf->num_sparticles * conf->num_dims / 4);

    launch_particle_init_kernel_common(
        conf,
        &(kernels[ALG_NAME_CAPS(PARTICLE_INIT_VEC_KERNEL)]),
        bufs,
        dev,
        (cl_uint *) &work_items_needed,
        NULL,
        "particle_init_vec"
        );
}

void launch_particle_init_kernel_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
    device *dev,
    cl_uint *global_worksize,
    cl_uint *local_worksize,
    char *kernel_label
    )
{
    #if LAUNCH_WARNINGS
    printf("Launching %s kernel.\n", kernel_label);
    printf("global_work_size: %u\n", global_worksize == NULL ? 0 : *global_worksize);
    printf("local_work_size: %u\n", local_worksize == NULL ? 0 : *local_worksize);
    #endif
    
    set_particle_init_kernel_args_common(
        conf,
        kernel,
        bufs
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        *kernel,
        1,
        NULL,
        global_worksize,
        local_worksize,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching %s kernel.", kernel_label);

    #if LAUNCH_WARNINGS
    printf("Done %s kernel launch.\n", kernel_label);
    #endif
}
