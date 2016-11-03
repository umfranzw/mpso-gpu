#include "drivers/tm/swap_particles/swap_particles_kernel_driver_tm.h"

void set_swap_particles_vec_kernel_args_tm(
    config_tm *conf,
    cl_kernel *kernel,
    mpso_bufs_tm *bufs,
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

/* void set_swap_particles_alt_kernel_args_tm( */
/*     config_tm *conf, */
/*     cl_kernel *kernel, */
/*     mpso_bufs_tm *bufs */
/*     ) */
/* { */
/*     cl_int error; */
/*     cl_uint arg_index = 0; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->best_indices_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index); */
/*     arg_index++; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->worst_indices_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index); */
/*     arg_index++; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->positions_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index); */
/*     arg_index++; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->velocities_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index); */
/*     arg_index++; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->pbest_fitnesses_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index); */
/*     arg_index++; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->pbest_positions_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles kernel arg %d\n", arg_index); */
/*     arg_index++; */
/* } */

/* void set_swap_particles_unvec_kernel_args_tm( */
/*     config_tm *conf, */
/*     cl_kernel *kernel, */
/*     mpso_bufs_tm *bufs */
/*     ) */
/* { */
/*     cl_int error; */
/*     cl_uint arg_index = 0; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->best_indices_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles_unvec kernel arg %d\n", arg_index); */
/*     arg_index++; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->worst_indices_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles_unvec kernel arg %d\n", arg_index); */
/*     arg_index++; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->positions_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles_unvec kernel arg %d\n", arg_index); */
/*     arg_index++; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->velocities_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles_unvec kernel arg %d\n", arg_index); */
/*     arg_index++; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->pbest_fitnesses_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles_unvec kernel arg %d\n", arg_index); */
/*     arg_index++; */

/*     error = clSetKernelArg( */
/*         *kernel, */
/*         arg_index, */
/*         sizeof(cl_mem), */
/*         &(bufs->pbest_positions_buf) */
/*         ); */
/*     check_error(error, "Error setting swap_particles_unvec kernel arg %d\n", arg_index); */
/*     arg_index++; */
/* } */

void launch_swap_particles_kernel_tm(
    config_tm *conf,
    cl_kernel *kernel,
    mpso_bufs_tm *bufs,
    cl_uint iter_num,
    device *dev,
    size_t *global_work_size,
    size_t *local_work_size,
    char *kernel_label,
    cl_uint combined
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

void launch_swap_particles_vec_kernel_tm(
    config_tm *conf,
    cl_kernel *kernels,
    mpso_bufs_tm *bufs,
    cl_uint iter_num,
    device *dev,
    cl_uint combined
    )
{
    static cl_uint swarms_per_group;
    static size_t local_work_size;
    static size_t global_work_size;
    
    if (iter_num == conf->exchange_iters)
    {
        swarms_per_group = dev->max_workgroup_size / conf->num_exchange;

        //check local memory constraint (2 buffers of size swarms_per_group * num_exchange * sizeof(cl_uint))
        cl_uint total_local_mem = swarms_per_group * conf->num_exchange * sizeof(cl_uint) * 2;
        if (total_local_mem > GPU_SHARED_MEM_LIMIT)
        {
            //scale back
            swarms_per_group = GPU_SHARED_MEM_LIMIT / (conf->num_exchange * sizeof(cl_uint) * 2);
            if (swarms_per_group == 0)
            {
                printf("Swap particles kernel driver unable to calculate local work size.\n");
                exit(1);
            }
        }
        local_work_size = (conf->num_swarms < swarms_per_group ? conf->num_swarms : swarms_per_group) * conf->num_exchange;
        global_work_size = local_work_size * (conf->num_swarms / swarms_per_group > 0 ? conf->num_swarms / swarms_per_group : 1);
    }
    
    set_swap_particles_vec_kernel_args_tm(
        conf,
        &(kernels[SWAP_PARTICLES_VEC_KERNEL_TM]),
        bufs,
        (cl_uint) swarms_per_group
        );

    launch_swap_particles_kernel_tm(
        conf,
        &(kernels[SWAP_PARTICLES_VEC_KERNEL_TM]),
        bufs,
        iter_num,
        dev,
        &global_work_size,
        &local_work_size,
        "swap_particles_vec",
        combined
        );
}

/* void launch_swap_particles_alt_kernel_tm( */
/*     config_tm *conf, */
/*     cl_kernel *kernels, */
/*     mpso_bufs_tm *bufs, */
/*     mpso_events_tm *events, */
/*     cl_uint iter_num, */
/*     device *dev, */
/*     cl_uint combined */
/*     ) */
/* { */
/*     size_t global_work_size = conf->num_swarms * conf->num_exchange * conf->num_dims / 4; */

/*     set_swap_particles_alt_kernel_args_tm( */
/*         conf, */
/*         &(kernels[SWAP_PARTICLES_ALT_KERNEL_TM]), */
/*         bufs */
/*         ); */

/*     launch_swap_particles_kernel_tm( */
/*         conf, */
/*         &(kernels[SWAP_PARTICLES_ALT_KERNEL_TM]), */
/*         bufs, */
/*         events, */
/*         iter_num, */
/*         dev, */
/*         &global_work_size, */
/*         NULL, */
/*         "swap_particles_alt", */
/*         combined */
/*         ); */
/* } */

/* void launch_swap_particles_unvec_kernel_tm( */
/*     config_tm *conf, */
/*     cl_kernel *kernels, */
/*     mpso_bufs_tm *bufs, */
/*     mpso_events_tm *events, */
/*     cl_uint iter_num, */
/*     device *dev, */
/*     cl_uint combined */
/*     ) */
/* { */
/*     size_t global_work_size = conf->num_swarms * conf->num_exchange * conf->num_dims; */

/*     set_swap_particles_unvec_kernel_args_tm( */
/*         conf, */
/*         &(kernels[SWAP_PARTICLES_UNVEC_KERNEL_TM]), */
/*         bufs */
/*         ); */

/*     launch_swap_particles_kernel_tm( */
/*         conf, */
/*         &(kernels[SWAP_PARTICLES_UNVEC_KERNEL_TM]), */
/*         bufs, */
/*         events, */
/*         iter_num, */
/*         dev, */
/*         &global_work_size, */
/*         NULL, */
/*         "swap_particles_vec", */
/*         combined */
/*         ); */
/* } */
