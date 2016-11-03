#include "drivers/ga/mut_restore/mut_restore_kernel_driver_ga.h"

void set_mut_restore_kernel_args_ga(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint particles_per_group
    )
{
    cl_int error;
    cl_uint arg_index = 0;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pre_mut_fit_buf)
        );
    check_error(error, "Error setting mut_restore kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pre_mut_pos_buf)
        );
    check_error(error, "Error setting mut_restore kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pre_mut_vel_buf)
        );
    check_error(error, "Error setting mut_restore kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->fitnesses_buf)
        );
    check_error(error, "Error setting mut_restore kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->positions_buf)
        );
    check_error(error, "Error setting mut_restore kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->velocities_buf)
        );
    check_error(error, "Error setting mut_restore kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->swarm_health_buf)
        );
    check_error(error, "Error setting mut_restore kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        particles_per_group * sizeof(cl_int),
        NULL
        );
    check_error(error, "Error setting mut_restore kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting mut_restore kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting mut_restore kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting mut_restore kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->unhealthy_iters)
        );
    check_error(error, "Error setting mut_restore kernel arg %d\n", arg_index);
}

void launch_mut_restore_kernel_ga(
    config_ga *conf,
    cl_kernel *kernels,
    mpso_bufs_ga *bufs,
    cl_uint iter_index,
    device *dev
    )
{
    static size_t local_work_size;
    static size_t global_work_size;
    static size_t particles_per_group;

    if (iter_index == 1) //this kernel is launched for the first time on the second iteration
    {
        //figure out launch dimensions
        const cl_uint threads_per_particle = conf->num_dims / 4;

        /* particles_per_group = conf->num_swarms * conf->num_sparticles < dev->max_workgroup_size / threads_per_particle ? conf->num_swarms * conf->num_sparticles : dev->max_workgroup_size / threads_per_particle; */

        /* local_work_size = particles_per_group * threads_per_particle;         */

        /* size_t num_groups = (conf->num_swarms * conf->num_sparticles) / particles_per_group + ((conf->num_swarms * conf->num_sparticles) % particles_per_group ? 1 : 0); */
        /* global_work_size = num_groups * local_work_size; */

        cl_uint local_mem_per_particle = 1;
        global_work_size = conf->num_swarms * conf->num_sparticles * threads_per_particle;
        local_work_size = calc_local_size(
            global_work_size,
            threads_per_particle,
            local_mem_per_particle * sizeof(cl_int),
            dev
            );

        particles_per_group = local_work_size / threads_per_particle;
    }

    #if LAUNCH_WARNINGS
    printf("Launching mut_restore kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: %u\n", local_work_size);
    #endif

    set_mut_restore_kernel_args_ga(
        conf,
        &(kernels[MUT_RESTORE_VEC_KERNEL_GA]),
        bufs,
        particles_per_group
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[MUT_RESTORE_VEC_KERNEL_GA],
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching mut_restore kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
