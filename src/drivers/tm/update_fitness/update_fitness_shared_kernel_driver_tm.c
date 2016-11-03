#include "drivers/tm/update_fitness/update_fitness_shared_kernel_driver_tm.h"

void set_update_fitness_shared_kernel_args_tm(
    config_tm *conf,
    cl_kernel *kernel,
    mpso_bufs_tm *bufs,
    cl_uint swarms_per_group
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
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->fitnesses_buf)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        //note: this is total size per workgroup
        swarms_per_group * conf->num_sparticles * conf->num_machines * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->etc_buf)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &swarms_per_group
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_machines)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting update fitness shared kernel arg %d\n", arg_index);
}

void launch_update_fitness_shared_kernel_tm(
    config_tm *conf,
    cl_kernel *kernel,
    mpso_bufs_tm *bufs,
    cl_uint iter_num,
    cl_uint swapped_last_iter,
    device *dev,
    char *kernel_label,
    cl_uint combined
    )
{
    static cl_uint swarms_per_group = 0;
    static size_t local_work_size;

    if (!iter_num)
    {
        if (conf->num_swarms * conf->num_sparticles <= dev->max_workgroup_size)
        {
            swarms_per_group = conf->num_swarms;
        }
        else
        {
            swarms_per_group = dev->max_workgroup_size / conf->num_sparticles;
        }

        if (swarms_per_group == 0)
        {
            printf("Update fitness shared kernel driver unable to determine local work size.\n");
        }

        //check local mem constraint
        cl_uint total_local_mem = swarms_per_group * conf->num_sparticles * conf->num_machines * sizeof(cl_float);
        if (total_local_mem > GPU_SHARED_MEM_LIMIT)
        {
            swarms_per_group = GPU_SHARED_MEM_LIMIT / (conf->num_machines * sizeof(cl_float));
        }
        
        local_work_size = swarms_per_group * conf->num_sparticles;
    }
    
    size_t global_work_size = local_work_size * ((conf->num_swarms / swarms_per_group) + (conf->num_swarms % swarms_per_group > 0 ? 1 : 0));

    #if LAUNCH_WARNINGS
    printf("Launching %s kernel.\n", kernel_label);
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: %u\n", local_work_size);
    printf("swarms_per_group: %u\n", swarms_per_group);
    printf("total local mem per CU (KB): %0.2f\n", swarms_per_group * conf->num_sparticles * conf->num_machines * sizeof(cl_float) / 1024.0);
    printf("total constant mem (KB): %0.2f\n", conf->num_machines * conf->num_dims * sizeof(cl_float) / 1024.0);
    #endif

    set_update_fitness_shared_kernel_args_tm(
        conf,
        kernel,
        bufs,
        swarms_per_group
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        *kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching %s kernel.", kernel_label);

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}

void launch_update_fitness_shared_vec_kernel_tm(
    config_tm *conf,
    cl_kernel *kernels,
    mpso_bufs_tm *bufs,
    cl_uint iter_num,
    cl_uint swapped_last_iter,
    device *dev,
    cl_uint combined
    )
{
    launch_update_fitness_shared_kernel_tm(
        conf,
        &(kernels[UPDATE_FITNESS_SHARED_VEC_KERNEL_TM]),
        bufs,
        iter_num,
        swapped_last_iter,
        dev,
        "update_fitness_shared_vec",
        combined
        );
}

/* void launch_update_fitness_shared_unvec_kernel_tm( */
/*     config_tm *conf, */
/*     cl_kernel *kernels, */
/*     mpso_bufs_tm *bufs, */
/*     mpso_events_tm *events, */
/*     cl_uint iter_num, */
/*     cl_uint swapped_last_iter, */
/*     device *dev, */
/*     cl_uint combined */
/*     ) */
/* { */
/*     //launch with same number of threads as vec version */
/*     launch_update_fitness_shared_kernel_tm( */
/*         conf, */
/*         &(kernels[UPDATE_FITNESS_SHARED_UNVEC_KERNEL_TM]), */
/*         bufs, */
/*         events, */
/*         iter_num, */
/*         swapped_last_iter, */
/*         dev, */
/*         "update_fitness_shared_unvec", */
/*         combined */
/*         ); */
/* } */
