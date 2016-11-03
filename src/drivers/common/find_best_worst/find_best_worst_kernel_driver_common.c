#include "drivers/common/find_best_worst/find_best_worst_kernel_driver_common.h"

void launch_find_best_worst_kernel_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
    device *dev,
    size_t *global_work_size,
    size_t *local_work_size,
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

void set_find_best_worst_alt_kernel_args_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
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
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        //note: this is total size per workgroup
        conf->num_sparticles * swarms_per_group * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->worst_indices_buf)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->best_indices_buf)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_exchange)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
}

void launch_find_best_worst_alt_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_num,
    device *dev
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;
    ALG_NAME(mpso_bufs) *bufs = (ALG_NAME(mpso_bufs) *) generic_bufs;

    //size_t local_work_size = conf->num_sparticles / 4;
    //size_t global_work_size = local_work_size * conf->num_swarms;
    static size_t local_work_size = 0;
    static cl_uint swarms_per_group = 0;
    
    if (iter_num == conf->exchange_iters)
    {
        cl_uint group_packing = dev->max_workgroup_size / (conf->num_sparticles / 4) +
            (dev->max_workgroup_size % (conf->num_sparticles / 4) > 0 ? 1 : 0);
        swarms_per_group = group_packing > conf->num_swarms ? conf->num_swarms : group_packing;
        if (swarms_per_group * (conf->num_sparticles / 4) > dev->max_workgroup_size)
        {
            if (swarms_per_group > 1)
            {
                swarms_per_group -= 1;
            }
            else
            {
                printf("Find best worst kernel driver unable to calculate local work size.\n");
                exit(0);
            }
        }

        //check local memory constraint (1 buffer of size swarms_per_group * num_sparticles * sizeof(cl_float))
        cl_uint total_local_mem = swarms_per_group * conf->num_sparticles * sizeof(cl_float);
        if (total_local_mem > GPU_SHARED_MEM_LIMIT)
        {
            //scale back
            swarms_per_group = GPU_SHARED_MEM_LIMIT / (conf->num_sparticles * sizeof(cl_float));
            if (swarms_per_group == 0)
            {
                printf("Find best worst kernel driver unable to calculate local work size.\n");
                exit(1);
            }
        }

        local_work_size = swarms_per_group * (conf->num_sparticles / 4);
    }
    size_t global_work_size = (conf->num_swarms / swarms_per_group) * local_work_size;

    set_find_best_worst_alt_kernel_args_common(
        conf,
        &(kernels[ALG_NAME_CAPS(FIND_BEST_WORST_ALT_KERNEL)]),
        bufs,
        swarms_per_group
        );

    launch_find_best_worst_kernel_common(
        conf,
        &(kernels[ALG_NAME_CAPS(FIND_BEST_WORST_ALT_KERNEL)]),
        bufs,
        dev,
        &global_work_size,
        &local_work_size,
        "find_best_worst_alt"
        );
}

void launch_find_best_worst_alt2_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_num,
    device *dev
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;
    ALG_NAME(mpso_bufs) *bufs = (ALG_NAME(mpso_bufs) *) generic_bufs;

    static size_t local_work_size = 0;
    static cl_uint swarms_per_group = 0;
    static size_t global_work_size = 0;
    
    if (iter_num + SWAP_OFFSET == conf->exchange_iters)
    {
        global_work_size = conf->num_swarms * (conf->num_sparticles / 2);
        local_work_size = calc_local_size(
            global_work_size,
            (conf->num_sparticles / 2),
            conf->num_sparticles * sizeof(cl_float),
            dev
            );
        swarms_per_group = local_work_size / (conf->num_sparticles / 2);

        /* swarms_per_group = dev->max_workgroup_size / (conf->num_sparticles / 2); */

        /* if (swarms_per_group * (conf->num_sparticles / 2) > dev->max_workgroup_size) */
        /* { */
        /*     if (swarms_per_group > 1) */
        /*     { */
        /*         swarms_per_group -= 1; */
        /*     } */
        /*     else */
        /*     { */
        /*         printf("Find best worst kernel driver unable to calculate local work size.\n"); */
        /*         exit(0); */
        /*     } */
        /* } */

        /* //check local memory constraint (1 buffer of size swarms_per_group * num_sparticles * sizeof(cl_float)) */
        /* cl_uint total_local_mem = swarms_per_group * conf->num_sparticles * sizeof(cl_float); */
        /* if (total_local_mem > GPU_SHARED_MEM_LIMIT) */
        /* { */
        /*     //scale back */
        /*     swarms_per_group = GPU_SHARED_MEM_LIMIT / (conf->num_sparticles * sizeof(cl_float)); */
        /*     if (swarms_per_group == 0) */
        /*     { */
        /*         printf("Find best worst kernel driver unable to calculate local work size.\n"); */
        /*         exit(1); */
        /*     } */
        /* } */
        
        /* local_work_size = swarms_per_group * (conf->num_sparticles / 2); */
        /* global_work_size = ((conf->num_swarms / swarms_per_group) > 0 ? (conf->num_swarms / swarms_per_group) : 1) * local_work_size; */
    }
    

    #if LAUNCH_WARNINGS
    printf("Launching find_best_worst_alt2 kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: %u\n", local_work_size);
    printf("swarms_per_group: %u\n", swarms_per_group);
    #endif

    set_find_best_worst_alt_kernel_args_common(
        conf,
        &(kernels[ALG_NAME_CAPS(FIND_BEST_WORST_ALT2_KERNEL)]),
        bufs,
        swarms_per_group
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[ALG_NAME_CAPS(FIND_BEST_WORST_ALT2_KERNEL)],
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching find_best_worst_alt2 kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif    
}

void set_find_best_worst_vec2_kernel_args_common(
    ALG_NAME(config) *conf,
    cl_kernel *kernel,
    ALG_NAME(mpso_bufs) *bufs,
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
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        //note: this is total size per workgroup
        swarms_per_group * conf->num_sparticles * 2 * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        //note: this is total size per workgroup
        swarms_per_group * conf->num_sparticles * 2 * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        //note: this is total size per workgroup
        swarms_per_group * conf->num_sparticles * 2 * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->worst_indices_buf)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->best_indices_buf)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_exchange)
        );
    check_error(error, "Error setting find_best_worst kernel arg %d\n", arg_index);
}

void launch_find_best_worst_vec2_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_num,
    device *dev
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;
    ALG_NAME(mpso_bufs) *bufs = (ALG_NAME(mpso_bufs) *) generic_bufs;

    //size_t local_work_size = make_multiple(conf->num_sparticles / 2, WAVEFRONT_SIZE);
    static size_t local_work_size = 0;
    static cl_uint swarms_per_group = 0;
    static size_t global_work_size = 0;
    
    if (iter_num == conf->exchange_iters) //do this the first time this kernel is launched (for the current rep)
    {
        cl_uint group_packing = dev->max_workgroup_size / (conf->num_sparticles / 2) +
            (dev->max_workgroup_size % (conf->num_sparticles / 2) > 0 ? 1 : 0);
        swarms_per_group = group_packing > conf->num_swarms ? conf->num_swarms : group_packing;

        //check dev->max_workgroup_size constraint
        if (swarms_per_group * (conf->num_sparticles / 2) > dev->max_workgroup_size)
        {
            if (swarms_per_group > 1)
            {
                swarms_per_group -= 1;
            }
            else
            {
                printf("Find best worst kernel driver unable to calculate local work size.\n");
                exit(1);
            }
        }

        //check local memory constraint (3 buffers of size swarms_per_group * num_sparticles * 2 * sizeof(cl_float))
        cl_uint total_local_mem = (swarms_per_group * conf->num_sparticles * 2 * sizeof(cl_float)) * 3;
        if (total_local_mem > GPU_SHARED_MEM_LIMIT)
        {
            //scale back
            swarms_per_group = GPU_SHARED_MEM_LIMIT / (conf->num_sparticles * 2 * sizeof(cl_float) * 3);
            if (swarms_per_group == 0)
            {
                printf("Find best worst kernel driver unable to calculate local work size.\n");
                exit(1);
            }
        }
        
        local_work_size = swarms_per_group * (conf->num_sparticles / 2);
        global_work_size = (conf->num_swarms / swarms_per_group) * local_work_size;
    }

    set_find_best_worst_vec2_kernel_args_common(
        conf,
        &(kernels[ALG_NAME_CAPS(FIND_BEST_WORST_VEC2_KERNEL)]),
        bufs,
        swarms_per_group
        );

    launch_find_best_worst_kernel_common(
        conf,
        &(kernels[ALG_NAME_CAPS(FIND_BEST_WORST_VEC2_KERNEL)]),
        bufs,
        dev,
        &global_work_size,
        &local_work_size,
        "find_best_worst_vec2"
        );
}
