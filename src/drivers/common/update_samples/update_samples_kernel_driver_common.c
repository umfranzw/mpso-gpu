#include "drivers/common/update_samples/update_samples_kernel_driver_common.h"

void set_update_samples_vec_kernel_driver_common_args(
    cl_kernel *kernel,
    cl_mem *samples_buf,
    cl_mem *src_buf,
    cl_uint num_samples,
    cl_uint sample_index,
    cl_uint init_output,
    cl_float divisor
    )
{
    cl_uint arg_index = 0;
    cl_int error;
    
    error = clSetKernelArg( 
        *kernel,
        arg_index,
        sizeof(cl_mem),
        samples_buf
        );
    check_error(error, "Error setting update_samples_common kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg( 
        *kernel,
        arg_index,
        sizeof(cl_mem),
        src_buf
        );
    check_error(error, "Error setting update_samples_common kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg( 
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &num_samples
        );
    check_error(error, "Error setting update_samples_common kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg( 
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &sample_index
        );
    check_error(error, "Error setting update_samples_common kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg( 
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &init_output
        );
    check_error(error, "Error setting update_samples_common kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg( 
        *kernel,
        arg_index,
        sizeof(cl_float),
        &divisor
        );
    check_error(error, "Error setting update_samples_common kernel arg %d", arg_index);
}

void launch_update_samples_vec_kernel_common(
    cl_kernel *kernels,
    cl_mem *samples_buf,
    cl_mem *src_buf,
    cl_uint num_samples,
    cl_uint sample_index,
    cl_uint init_output,
    cl_float divisor,
    device *dev
    )
{
    cl_uint global_work_size;
    if (divisor > 0)
    {
        //max(num_samples / 4, num_samples % 4);
        global_work_size = num_samples / 4 > num_samples % 4 ? num_samples / 4 : num_samples % 4;
    }
    else
    {
        //only one thread needed for update of sample_index
        global_work_size = 1;
    }

    #if LAUNCH_WARNINGS
    printf("Launching update_samples_common kernel.\n");
    printf("global_work_size: %u\n", global_work_size);
    printf("local_work_size: NULL\n");
    #endif

    set_update_samples_vec_kernel_driver_common_args(
        &(kernels[ALG_NAME_CAPS(UPDATE_SAMPLES_VEC_KERNEL)]),
        samples_buf,
        src_buf,
        num_samples,
        sample_index,
        init_output,
        divisor
        );
    
    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[ALG_NAME_CAPS(UPDATE_SAMPLES_VEC_KERNEL)],
        1,
        NULL,
        &global_work_size,
        NULL,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching update_samples_common kernel.");

    #if LAUNCH_WARNINGS
    printf("Done update_samples_common kernel launch.\n");
    #endif
}
