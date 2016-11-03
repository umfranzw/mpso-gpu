#ifndef _CLHELPER_H_
#define _CLHELPER_H_

#include <stdio.h>
#include "CL/cl.h"
#include "global_constants.h"
#include "utils.h"
#include "devices.h"
#include ALG_HEADER_STR(kernels)

//prototype declarations
void init_opencl(
    char *kernel_filename,
    cl_program *program,
    cl_kernel *kernel_buf,
    device *cpu,
    device *gpu
    );

void build_program(
    char *filename,
    cl_program *program,
    device *cpu,
    device *gpu
    );

void shutdown_opencl(
    cl_program *program,
    cl_kernel *kernels,
    device *cpu,
    device *gpu
    );

#endif
