#include "drivers/common/update_fitness/f5_kernel_driver_common.h"

void set_f5_kernel_args(
    ALG_NAME_COMMON(config) *conf,
    cl_kernel *kernel,
    ALG_NAME_COMMON(mpso_bufs) *bufs,
    cl_uint particles_per_group,
    cl_uint threads_per_particle,
    cl_uint scratch_per_particle
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
    check_error(error, "Error setting f5 kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->fitnesses_buf)
        );
    check_error(error, "Error setting f5 kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->optimum_buf)
        );
    check_error(error, "Error setting f5 kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->rot_matrix_buf)
        );
    check_error(error, "Error setting f5 kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->perm_vec_buf)
        );
    check_error(error, "Error setting f5 kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        particles_per_group * scratch_per_particle * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting f5 kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting f5 kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting f5 kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting f5 kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->m)
        );
    check_error(error, "Error setting f5 kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &threads_per_particle
        );
    check_error(error, "Error setting f5 kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &scratch_per_particle
        );
    check_error(error, "Error setting f5 kernel arg %d\n", arg_index);
}

void launch_f5_kernel(
    ALG_NAME_COMMON(config) *conf,
    cl_kernel *kernels,
    ALG_NAME_COMMON(mpso_bufs) *bufs,
    cl_uint iter_index,
    device *dev
    )
{
    static size_t local_work_size;
    static size_t global_work_size;
    static size_t particles_per_group;
    static size_t local_mem_per_particle;

    if (iter_index == 0)
    {
        //figure out launch dimensions

        //Permutations require d / 4 threads, rotation requires at least m / 4 threads, function1 requires m / 4 threads, function2 requires (d - m) / 4 threads.
        const cl_uint threads_per_particle = conf->num_dims / 4;

        cl_uint workgroup_size = conf->num_swarms * conf->num_sparticles * threads_per_particle;

        if (workgroup_size > dev->max_workgroup_size)
        {
            workgroup_size = dev->max_workgroup_size;
        }
        local_work_size = workgroup_size - (workgroup_size % threads_per_particle);
        
        particles_per_group = local_work_size / threads_per_particle;

        size_t num_groups = (conf->num_swarms * conf->num_sparticles) / particles_per_group + ((conf->num_swarms * conf->num_sparticles) % particles_per_group ? 1 : 0);
        global_work_size = num_groups * local_work_size;

        //figure out local memory requirements
        const cl_uint mem_for_perm = conf->num_dims;
        const cl_uint mem_for_rot = conf->m * conf->m / 2;
        local_mem_per_particle = mem_for_perm < mem_for_rot ? mem_for_rot : mem_for_perm;
    }

    #if LAUNCH_WARNINGS
    printf("Launching F5 benchmark kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: %u\n", local_work_size);
    printf("particles_per_group: %u\n", particles_per_group);
    #endif

    set_f5_kernel_args(
        conf,
        &(kernels[ALG_NAME_COMMON_CAPS(F5_KERNEL)]),
        bufs,
        particles_per_group,
        local_work_size / particles_per_group,
        local_mem_per_particle
        );

    /* clWaitForEvents(events->wait_list_lens[ALG_NAME_COMMON_CAPS(UPDATE_FITNESS_EVENT)], events->wait_lists[ALG_NAME_COMMON_CAPS(UPDATE_FITNESS_EVENT)]); */
    /* printf("Launching! %u\n", iter_index); */

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[ALG_NAME_COMMON_CAPS(F5_KERNEL)],
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching F5 benchmark kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
