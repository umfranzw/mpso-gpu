#include "drivers/common/find_min/find_min_kernel_driver_common.h"

void set_find_min_kernel_driver_common_args(
    cl_kernel *kernel,
    cl_mem *input_buf,
    cl_uint input_len,
    cl_mem *global_scratch_buf,
    cl_mem *result_buf,
    cl_uint result_index,
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
        global_scratch_buf
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

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        result_buf
        );
    check_error(error, "Error setting find_min_common kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &result_index
        );
    check_error(error, "Error setting find_min_common kernel arg %d", arg_index);
}

void find_min_cpu(
    cl_mem *input_buf,
    cl_uint input_len,
    cl_mem *results_buf,
    cl_uint result_index,
    device *dev
    )
{
    #if LAUNCH_WARNINGS
    printf("Performing find_min_common cpu.\n");
    #endif

    cl_event map_events[2];
    cl_float *host_input_buf = (cl_float *) map_buffer(
        input_buf,
        CL_MAP_READ,
        sizeof(cl_float) * input_len,
        dev,
        &(map_events[0])
        );

    cl_int error;
    cl_float *host_results_buf = (cl_float *) clEnqueueMapBuffer(
        dev->cmd_q,
        *results_buf,
        CL_FALSE,
        CL_MAP_WRITE,
        //need only map one element (at an offset)
        sizeof(cl_float) * result_index,
        sizeof(cl_float),
        0,
        NULL,
        &(map_events[1]),
        &error
        );
    check_error(error, "Error mapping result buffer in find_min_cpu().");
    
    clWaitForEvents(2, map_events);
    clReleaseEvent(map_events[1]);
    clReleaseEvent(map_events[0]);

    cl_uint i;
    //we are assuming there is at least one element in the input array
    cl_float smallest = host_input_buf[0];
    for (i = 1; i < input_len; i++)
    {
        if (host_input_buf[i] < smallest)
        {
            smallest = host_input_buf[i];
        }
    }

    host_results_buf[0] = smallest; //indexing was already performed in the map buffer call
    
    unmap_buffer(
        input_buf,
        host_input_buf,
        dev,
        &(map_events[0])
        );
    unmap_buffer(
        results_buf,
        host_results_buf,
        dev,
        &(map_events[1])
        );
    clReleaseEvent(map_events[1]);
    clReleaseEvent(map_events[0]);
    
    #if LAUNCH_WARNINGS
    printf("Done performing find_min_common cpu.\n");
    #endif
}

void do_kernel_launch(
    cl_kernel *kernels,
    cl_mem *input_buf,
    cl_uint input_len,
    cl_mem *global_scratch_buf,
    cl_mem *result_buf,
    cl_uint result_index,
    device *dev
    )
{
    cl_uint global_work_size = input_len / 4;
    cl_uint local_mem_elements = input_len / 8;

    cl_uint local_work_size = dev->max_workgroup_size;

    if (global_work_size < local_work_size)
    {
        local_work_size = global_work_size;
    }
    else if (global_work_size % local_work_size)
    {
        //we need one extra, partially filled workgroup
        global_work_size += local_work_size;
    }
    
    #if LAUNCH_WARNINGS
    printf("Launching find_min_common kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: %u\n", local_work_size);
    printf("local_mem elements: %u\n", local_mem_elements);
    #endif

    set_find_min_kernel_driver_common_args(
        &(kernels[ALG_NAME_CAPS(FIND_MIN_VEC_KERNEL)]),
        input_buf,
        input_len,
        global_scratch_buf,
        result_buf,
        result_index,
        local_mem_elements
        );

    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[ALG_NAME_CAPS(FIND_MIN_VEC_KERNEL)],
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching find_min_common kernel.");

    #if LAUNCH_WARNINGS
    printf("Done find_min_common kernel launch.\n");
    #endif
}

void launch_find_min_vec_kernel_common(
    cl_kernel *kernels,
    cl_mem *input_buf,
    cl_uint input_len,
    cl_mem *global_scratch_buf,
    cl_mem *result_buf,
    cl_uint result_index,
    device *dev
    )
{
    if (input_len > 8)
    {
        do_kernel_launch(
            kernels,
            input_buf,
            input_len,
            global_scratch_buf,
            result_buf,
            result_index,
            dev
            );
    }
    else
    {
        find_min_cpu(
            input_buf,
            input_len,
            result_buf,
            result_index,
            dev
            );
    }
}
