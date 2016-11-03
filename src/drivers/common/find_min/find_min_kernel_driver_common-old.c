#include "drivers/common/find_min/find_min_kernel_driver_common.h"

void set_find_min_kernel_driver_common_args(
    cl_kernel *kernel,
    cl_mem *input_buf,
    cl_mem *output_buf,
    cl_uint input_len,
    cl_uint output_len,
    cl_uint output_result_index,
    cl_float divisor,
    cl_uint init_output,
    cl_uint local_mem_elements
    )
{
    cl_uint arg_index = 0;
    cl_int error;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        input_buf
        );
    check_error(error, "Error setting find_min_common kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        output_buf
        );
    check_error(error, "Error setting find_min_common kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        local_mem_elements * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting find_min_common kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &input_len
        );
    check_error(error, "Error setting find_min_common kernel arg %d", arg_index);
    arg_index++;

    //uint num_samples = conf->max_iters / conf->fitness_sample_interval;
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &output_len
        );
    check_error(error, "Error setting find_min_common kernel arg %d", arg_index);
    arg_index++;

    //uint sample_index = ((iter_index + 1) / conf->fitness_sample_interval) - 1;
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &output_result_index
        );
    check_error(error, "Error setting find_min_common kernel arg %d", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &init_output
        );
    check_error(error, "Error setting find_min_common kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float),
        &divisor
        );
    check_error(error, "Error setting find_min_common kernel arg %d", arg_index);
}

void launch_find_min_vec_kernel_common(
    cl_kernel *kernels,
    void *generic_events,
    cl_mem *input_buf,
    cl_mem *output_buf,
    cl_uint input_len,
    cl_uint output_len,
    cl_uint output_result_index,
    cl_float divisor,
    cl_uint init_output,
    cl_uint iter_index,
    cl_uint fitness_sample_interval,
    device *gpu
    )
{
    cl_uint global_work_size = input_len / 4;
    cl_uint local_mem_elements = input_len / 4;

    cl_uint local_work_size = global_work_size < GPU_WORKGROUP_SIZE ? global_work_size : GPU_WORKGROUP_SIZE;
    
    #if LAUNCH_WARNINGS
    printf("Launching find_min_common kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: NULL\n");
    printf("local_mem elements: %u\n", local_mem_elements);
    #endif

    ALG_NAME(mpso_events) *events = (ALG_NAME(mpso_events) *) generic_events;
    
    set_find_min_kernel_driver_common_args(
        &(kernels[ALG_NAME_CAPS(FIND_MIN_VEC_KERNEL)]),
        input_buf,
        output_buf,
        input_len,
        output_len,
        output_result_index,
        divisor,
        init_output,
        local_mem_elements
        );

    if (iter_index + 1 > fitness_sample_interval)
    {
        clReleaseEvent(events->completions[ALG_NAME_CAPS(FIND_MIN_EVENT)]);
    }

    if (ALG == ALG_GA)
    {
        events->wait_lists[ALG_NAME_CAPS(FIND_MIN_EVENT)][0] = events->completions[ALG_NAME_CAPS(UPDATE_FITNESS_EVENT)];
    }
    else
    {
        events->wait_lists[ALG_NAME_CAPS(FIND_MIN_EVENT)][0] = events->completions[ALG_NAME_CAPS(UPDATE_BEST_VALS_EVENT)];
    }
    events->wait_list_lens[ALG_NAME_CAPS(FIND_MIN_EVENT)] = 1;

    cl_int error = clEnqueueNDRangeKernel(
        gpu->cmd_q,
        kernels[ALG_NAME_CAPS(FIND_MIN_VEC_KERNEL)],
        1,
        NULL,
        &global_work_size,
        NULL,
        events->wait_list_lens[ALG_NAME_CAPS(FIND_MIN_EVENT)],
        events->wait_lists[ALG_NAME_CAPS(FIND_MIN_EVENT)],
        &(events->completions[ALG_NAME_CAPS(FIND_MIN_EVENT)])
        );
    check_error(error, "Error launching find_min_common kernel.");

    #if BLOCKING
    clWaitForEvents(1, &(events->completions[ALG_NAME_CAPS(FIND_MIN_EVENT)]));
    #endif
    
    #if LAUNCH_WARNINGS
    printf("Done find_min_common kernel launch.\n");
    #endif
}
