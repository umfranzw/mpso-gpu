#include "drivers/common/update_fitness/f1_kernel_driver_common.h"

void set_f1_kernel_args(
    ALG_NAME_COMMON(config) *conf,
    cl_kernel *kernel,
    ALG_NAME_COMMON(mpso_bufs) *bufs,
    cl_uint particles_per_group
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
    check_error(error, "Error setting f1 kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->fitnesses_buf)
        );
    check_error(error, "Error setting f1 kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        particles_per_group * conf->num_dims / 4 * 2 * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting f1 kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->optimum_buf)
        );
    check_error(error, "Error setting f1 kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting f1 kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting f1 kernel arg %u\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting f1 kernel arg %u\n", arg_index);
}

void launch_f1_kernel(
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

    if (iter_index == 0)
    {
        //figure out launch dimensions
        /* size_t workgroup_size = conf->num_swarms * conf->num_sparticles * conf->num_dims / 4 < dev->max_workgroup_size ? conf->num_swarms * conf->num_sparticles * conf->num_dims / 4 : dev->max_workgroup_size; */
        /* local_work_size = workgroup_size - (workgroup_size % (conf->num_dims / 4)); */
        /* particles_per_group = local_work_size / (conf->num_dims / 4); */

        /* size_t num_groups = (conf->num_swarms * conf->num_sparticles) / particles_per_group + ((conf->num_swarms * conf->num_sparticles) % particles_per_group ? 1 : 0); */
        /* global_work_size = num_groups * local_work_size; */

        global_work_size = conf->num_swarms * conf->num_sparticles * conf->num_dims / 4;
        local_work_size = calc_local_size(
            global_work_size,
            conf->num_dims / 4,
            conf->num_dims / 4 * 2 * sizeof(cl_float),
            dev
            );
        particles_per_group = local_work_size / (conf->num_dims / 4);
    }

    #if LAUNCH_WARNINGS
    printf("Launching F1 benchmark kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: %u\n", local_work_size);
    printf("particles_per_group: %u\n", particles_per_group);
    #endif

    set_f1_kernel_args(
        conf,
        &(kernels[ALG_NAME_COMMON_CAPS(F1_KERNEL)]),
        bufs,
        particles_per_group
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[ALG_NAME_COMMON_CAPS(F1_KERNEL)],
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching F1 benchmark kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
