#include "drivers/common/init_rot_matrix/init_rot_matrix_kernel_driver_common.h"

void set_init_rot_matrix_args_common(
    ALG_NAME_COMMON(config) *conf,
    cl_kernel *kernel,
    ALG_NAME_COMMON(mpso_bufs) *bufs,
    cl_uint width,
    cl_uint height,
    cl_uint block_size
    )
{
    cl_uint arg_index = 0;
    cl_int error;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->rot_matrix_buf)
        );
    check_error(error, "Error setting init_rot_matrix kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->initial_rot_matrix_buf)
        );
    check_error(error, "Error setting init_rot_matrix kernel arg %d", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        block_size * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting init_rot_matrix kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &width
        );
    check_error(error, "Error setting init_rot_matrix kernel arg %d", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &height
        );
    check_error(error, "Error setting init_rot_matrix kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &block_size
        );
    check_error(error, "Error setting init_rot_matrix kernel arg %d", arg_index);
}

void launch_init_rot_matrix_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    device *dev
    )
{
    ALG_NAME_COMMON(config) *conf = (ALG_NAME_COMMON(config) *) generic_conf;
    ALG_NAME_COMMON(mpso_bufs) *bufs = (ALG_NAME_COMMON(mpso_bufs) *) generic_bufs;

    size_t local_size[2] = {1, 1};
    cl_uint block_size =  local_size[0] * local_size[1] * 4;
    size_t global_size[2] = {conf->m / 4, conf->m / 4};

    #if LAUNCH_WARNINGS
    printf("Launching init_rot_matrix kernel.\n");
    printf("global_work_size: %u, %u\n", global_size[0], global_size[1]);
    printf("local_work_size: %u, %u\n", local_size[0], local_size[1]);
    #endif

    set_init_rot_matrix_args_common(
        conf,
        &(kernels[ALG_NAME_COMMON_CAPS(INIT_ROT_MATRIX_VEC_KERNEL)]),
        bufs,
        conf->m,
        conf->m,
        block_size
        );

    clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[ALG_NAME_COMMON_CAPS(INIT_ROT_MATRIX_VEC_KERNEL)],
        2,
        NULL,
        global_size,
        local_size,
        0, //no waiting needed here
        NULL,
        NULL
        );

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
