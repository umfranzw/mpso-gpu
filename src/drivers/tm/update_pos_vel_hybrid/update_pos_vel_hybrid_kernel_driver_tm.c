#include "drivers/tm/update_pos_vel_hybrid/update_pos_vel_hybrid_kernel_driver_tm.h"

void set_update_pos_vel_hybrid_kernel_args_tm(
    config_tm *conf,
    cl_kernel *kernel,
    mpso_bufs_tm *bufs,
    cl_uint iter_index,
    cl_uint group_size
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
    check_error(error, "Error setting update_pos_vel_hybrid kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->velocities_buf)
        );
    check_error(error, "Error setting update_pos_vel_hybrid kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->pbest_positions_buf)
        );
    check_error(error, "Error setting update_pos_vel_hybrid kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_mem),
        &(bufs->sbest_positions_buf)
        );
    check_error(error, "Error setting update_pos_vel_hybrid kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_float) * group_size * 4,
        NULL
        );
    check_error(error, "Error setting update_pos_vel_hybrid kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(iter_index)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_swarms)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;
    
    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_sparticles)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_dims)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->num_machines)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->omega)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->c1)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->c2)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
    arg_index++;

    error = clSetKernelArg(
        *kernel,
        arg_index,
        sizeof(cl_uint),
        &(conf->seed)
        );
    check_error(error, "Error setting update_pos_vel kernel arg %d\n", arg_index);
}

void launch_update_pos_vel_hybrid_kernel_tm(
    config_tm *conf,
    cl_kernel *kernels,
    mpso_bufs_tm *bufs,
    cl_uint iter_index,
    device *dev
    )
{
    size_t global_work_size = conf->num_swarms * conf->num_sparticles * conf->num_dims / 4;
    static size_t local_work_size = 0;

    if (iter_index == 0)
    {
        if (global_work_size < dev->max_workgroup_size)
        {
            local_work_size = global_work_size;
        }
        else
        {
            local_work_size = dev->max_workgroup_size;
            while (local_work_size > 0 &&
                   (global_work_size % local_work_size > 0 || //global work size must be multiple of local work size
                    sizeof(cl_float) * local_work_size * 4 > GPU_SHARED_MEM_LIMIT) //buffer size must not exceed available local mem
                )
            {
                local_work_size -= 4;
            }
            if (local_work_size == 0)
            {
                printf("update_pos_vel_hybrid kernel unable to determine local work size.\n");
                exit(1);
            }
        }
    }

    #if LAUNCH_WARNINGS
    printf("Launching update_pos_vel_hybrid kernel.\n");
    printf("global_work_size: %u\n", global_work_size == NULL ? 0 : global_work_size);
    printf("local_work_size: %u\n", local_work_size == NULL ? 0 : local_work_size);
    #endif

    set_update_pos_vel_hybrid_kernel_args_tm(
        conf,
        &(kernels[UPDATE_POS_VEL_HYBRID_VEC_KERNEL_TM]),
        bufs,
        iter_index,
        (cl_uint) local_work_size
        );
    
    cl_int error = clEnqueueNDRangeKernel(
        dev->cmd_q,
        kernels[UPDATE_POS_VEL_HYBRID_VEC_KERNEL_TM],
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
        );
    check_error(error, "Error launching update_pos_vel_hybrid kernel.");

    #if LAUNCH_WARNINGS
    printf("Done.\n");
    #endif
}
