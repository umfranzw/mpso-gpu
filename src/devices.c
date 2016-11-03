#include "devices.h"

//Detects a device of the specified type (CPU, GPU) and sets up an OpenCL device object.
void init_devices(
    cl_platform_id platform_id, 
    device *cpu,
    device *gpu
    )
{
    cl_int error;
    cl_device_id dev_ids_buf[2];

    //get the device ids
    error = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_CPU,
        1,
        dev_ids_buf,
        NULL
        );
    check_error(error, "Error getting CPU device id.");
    cpu->device_id = dev_ids_buf[0];

    print_device_name(
        "CPU",
        cpu
        );

    error = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_GPU,
        2,
        dev_ids_buf,
        NULL
        );
    check_error(error, "Error getting GPU device id.");
    gpu->device_id = dev_ids_buf[1];

    print_device_name(
        "GPU",
        gpu
        );

    //create a single context for both devices
    cl_context_properties context_props[3];
    context_props[0]= CL_CONTEXT_PLATFORM;
    context_props[1]= (cl_context_properties) platform_id;
    context_props[2]= 0;

    cl_context context;
    dev_ids_buf[0] = cpu->device_id;
    dev_ids_buf[1] = gpu->device_id;
    
    context = clCreateContext(
        context_props,
        2,
        dev_ids_buf,
        NULL,
        NULL,
        &error
        );    
    check_error(error, "Error creating device context.");
    cpu->context = context;
    gpu->context = context;

    //create a separate command queue for each of the devices
    //cl_command_queue_properties q_props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
    cpu->cmd_q = clCreateCommandQueue(
        cpu->context,
        cpu->device_id,
        NULL,
        &error
        );
    check_error(error, "Error creating CPU command queue.");

    gpu->cmd_q = clCreateCommandQueue(
        gpu->context,
        gpu->device_id,
        NULL,
        &error
        );
    check_error(error, "Error creating GPU command queue.");

    gpu->max_workgroup_size = GPU_WORKGROUP_SIZE;
    gpu->max_local_mem = GPU_SHARED_MEM_LIMIT;
    cpu->max_workgroup_size = CPU_WORKGROUP_SIZE;
    cpu->max_local_mem = UINT_MAX; //CPU has no local mem, set this so as to ensure the drivers never scale down because of local mem limits
}

void print_device_name(
    char *title,
    device *dev
    )
{
    //first get size of name string
    size_t name_size;
    clGetDeviceInfo(
        dev->device_id,
        CL_DEVICE_NAME,
        0,
        NULL,
        &name_size
        );
    
    //now get name
    cl_char *name = (cl_char *) malloc(name_size);
    clGetDeviceInfo(
        dev->device_id,
        CL_DEVICE_NAME,
        name_size,
        name,
        NULL
        );
    
    //now print
    fprintf(stderr, "%s: %s\n", title, name);

    free(name);
}
