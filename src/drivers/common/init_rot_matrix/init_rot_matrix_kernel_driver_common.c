#include "drivers/common/init_rot_matrix/init_rot_matrix_kernel_driver_common.h"

void set_init_rot_matrix_args_common(
    ALG_NAME_COMMON(config) *conf,
    cl_kernel *kernel,
    ALG_NAME_COMMON(mpso_bufs) *bufs,
    cl_uint local_mem
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
        local_mem * sizeof(cl_float),
        NULL
        );
    check_error(error, "Error setting init_rot_matrix kernel arg %d", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->m)
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

    //right now this only works for one work group (max matrix dimensions 1024 x 1024)
    size_t global_size = (conf->m / 4) * (conf->m / 4);
    
    cl_uint rows_per_group = 4;
    cl_uint local_mem = rows_per_group * conf->m;
    size_t local_size = conf->m / 4;

    #if LAUNCH_WARNINGS
    printf("Launching init_rot_matrix kernel.\n");
    printf("global_work_size: %u\n", global_size);
    printf("local_work_size: %u\n", local_size);
    printf("local_mem elements: %u\n", local_mem);
    #endif

    set_init_rot_matrix_args_common(
        conf,
        &(kernels[ALG_NAME_COMMON_CAPS(INIT_ROT_MATRIX_VEC_KERNEL)]),
        bufs,
        local_mem
        );

    clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[ALG_NAME_COMMON_CAPS(INIT_ROT_MATRIX_VEC_KERNEL)],
        1,
        NULL,
        &global_size,
        &local_size,
        0,
        NULL,
        NULL
        );

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
