#include "drivers/ga/update_best_vals/update_best_vals_kernel_driver_spacial.h"

void set_update_best_vals_vec_kernel_args_spacial(
    config_ga *conf,
    cl_kernel *kernel,
    mpso_bufs_ga *bufs,
    cl_uint swarms_per_group
    )
{
    cl_int error;
    cl_uint arg_index = 0;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->fitnesses_buf)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->positions_buf)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pbest_fitnesses_buf)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pbest_positions_buf)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->sbest_fitnesses_buf)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->sbest_positions_buf)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        conf->num_sparticles * swarms_per_group * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &(conf->unhealthy_ratio)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->alg_health_buf)
        );
    check_error(error, "Error setting update_best_vals kernel arg %d\n", arg_index);
}

void calc_update_best_vals_sizes_spacial(
    config_ga *conf,
    size_t *local_work_size,
    size_t *global_work_size,
    cl_uint *swarms_per_group,
    device *dev
    )
{
    if (conf->num_swarms * conf->num_sparticles <= dev->max_workgroup_size)
    {
        *swarms_per_group = conf->num_swarms;
    }
    else
    {
        *swarms_per_group = dev->max_workgroup_size / conf->num_sparticles;
    }
    
    //check local mem constraint
    cl_uint total_local_mem = conf->num_sparticles * (*swarms_per_group) * sizeof(cl_float);
    if (total_local_mem > GPU_SHARED_MEM_LIMIT)
    {
        *swarms_per_group = GPU_SHARED_MEM_LIMIT / (conf->num_sparticles * sizeof(cl_float));
    }
    
    *local_work_size = (*swarms_per_group) * conf->num_sparticles;
    *global_work_size = (*local_work_size) * ((conf->num_swarms / (*swarms_per_group)) + (conf->num_swarms % (*swarms_per_group) > 0 ? 1 : 0));
}    

void launch_update_best_vals_vec_kernel_spacial(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_num,
    device *dev
    )
{
    //Note: we can't launch this with more threads, since this will increase the number of workgroups, driving up local memory usage, driving down kernel occupancy. Instead, we have to iterate inside the kernel.
    config_ga *conf = (config_ga *) generic_conf;
    mpso_bufs_ga *bufs = (mpso_bufs_ga *) generic_bufs;

    static size_t local_work_size;
    static size_t global_work_size;
    static cl_uint swarms_per_group;

    if (iter_num == 0)
    {
        /* calc_update_best_vals_sizes_spacial( */
        /*     conf, */
        /*     &local_work_size, */
        /*     &global_work_size, */
        /*     &swarms_per_group, */
        /*     dev */
        /*     ); */
        global_work_size = conf->num_swarms * conf->num_sparticles;
        local_work_size = calc_local_size(
            global_work_size,
            conf->num_sparticles,
            conf->num_sparticles * sizeof(cl_float),
            dev
            );
        swarms_per_group = local_work_size / conf->num_sparticles;
    }
     
    #if LAUNCH_WARNINGS
    printf("Launching update_best_vals_vec_spacial kernel.\n");
    printf("global_work_size: %u\n", global_work_size == NULL ? 0 : global_work_size);
    printf("local_work_size: %u\n", local_work_size == NULL ? 0 : local_work_size);
    printf("swarms_per_group: %u\n", swarms_per_group);
    #endif

    set_update_best_vals_vec_kernel_args_spacial(
        conf,
        &(kernels[UPDATE_BEST_VALS_VEC_KERNEL_GA]),
        bufs,
        swarms_per_group
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[UPDATE_BEST_VALS_VEC_KERNEL_GA],
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching update_best_vals kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
