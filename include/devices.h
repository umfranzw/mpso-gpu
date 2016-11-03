#ifndef _DEVICES_H_
#define _DEVICES_H_

#include "CL/cl.h"
#include "utils.h"

/******************
 * GPU/CPU Hardware Parameters *
 ******************/
//device-specific max workgroup size
#define GPU_WORKGROUP_SIZE 256
#define CPU_WORKGROUP_SIZE 1024

#define WAVEFRONT_SIZE 64

//bytes
#define GPU_MAX_BUF_SIZE 142 * 1024 * 1024

//(in bytes) 32 KB
#define GPU_SHARED_MEM_LIMIT 32768

//holds come handy parameters associated with OpenCL devices
typedef struct device
{
    cl_device_id device_id;
    cl_context context;
    cl_command_queue cmd_q;
    cl_uint max_workgroup_size; //set to one of the two constants given above
    cl_uint max_local_mem;
} device;

void init_devices(
    cl_platform_id platform_id,
    device *cpu,
    device *gpu
    );

void print_device_name(
    char *title,
    device *dev
    );

#endif
