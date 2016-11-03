#include "drivers/common/swap_particles/swap_particles_kernel_driver_common.h"

void set_swap_particles_vec_kernel_args_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
    cl_uint swarms_per_group
    )
{
    cl_int error;
    cl_uint arg_index = 0;

    /* //for scratch */
    /* error = clSetKernelArg( */
    /*     *kernel, */
    /*     arg_index, */
    /*     conf->num_exchange * swarms_per_group * sizeof(cl_uint), */
    /*     NULL */
    /*     ); */
    /* check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index); */
    /* arg_index++; */

    /* //for scratch */
    /* error = clSetKernelArg( */
    /*     *kernel, */
    /*     arg_index, */
    /*     conf->num_exchange * swarms_per_group * sizeof(cl_uint), */
    /*     NULL */
    /*     ); */
    /* check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index); */
    /* arg_index++; */

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->best_indices_buf)
        );
    check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->worst_indices_buf)
        );
    check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->positions_buf)
        );
    check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->velocities_buf)
        );
    check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pbest_fitnesses_buf)
        );
    check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pbest_positions_buf)
        );
    check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_exchange)
        );
    check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index);
}

void launch_swap_particles_vec_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_num,
    device *dev
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;
    ALG_NAME(mpso_bufs) *bufs = (ALG_NAME(mpso_bufs) *) generic_bufs;

    static cl_uint swarms_per_group;
    static size_t local_work_size;
    static size_t global_work_size;
    
    if (iter_num + SWAP_OFFSET == conf->exchange_iters)
    {
        /* swarms_per_group = dev->max_workgroup_size / conf->num_exchange; */

        /* local_work_size = (conf->num_swarms < swarms_per_group ? conf->num_swarms : swarms_per_group) * conf->num_exchange; */
        /* global_work_size = local_work_size * (conf->num_swarms / swarms_per_group > 0 ? conf->num_swarms / swarms_per_group : 1); */

        global_work_size = conf->num_swarms * conf->num_exchange;
        local_work_size = calc_local_size(
            global_work_size,
            conf->num_exchange,
            0,
            dev
            );
        swarms_per_group = local_work_size / conf->num_exchange;
    }
    
    #if LAUNCH_WARNINGS
    printf("Launching swap_particles_vec_common kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: %u\n", local_work_size == NULL ? 0 : local_work_size);
    printf("swarms_per_group: %u\n", swarms_per_group);
    #endif
    
    set_swap_particles_vec_kernel_args_common(
        conf,
        &(kernels[ALG_NAME_CAPS(SWAP_PARTICLES_VEC_KERNEL)]),
        bufs,
        (cl_uint) swarms_per_group
        );
   
    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[ALG_NAME_CAPS(SWAP_PARTICLES_VEC_KERNEL)],
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching swap_particles_vec_common kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
