#include "clhelper.h"

//Initializes OpenCL
void init_opencl(
    char *kernel_filename,
    cl_program *program,
    cl_kernel *kernel_buf,
    device *cpu,
    device *gpu
    )
{
    cl_int error;
    
    //get the id of the platform
    cl_platform_id platform_id;
    cl_uint num_platforms;

    //get number of available platforms
    error = clGetPlatformIDs(
        1,
        &platform_id,
        &num_platforms
        );
    check_error(error, "Error getting platform id.");

    //set up CPU and GPU devices
    init_devices(
        platform_id,
        cpu,
        gpu
        );

    build_program(
        kernel_filename,
        program,
        cpu,
        gpu
        );

    //create the kernel objects
    //Note: apparently the AMD OpenCL implementation inserts them in alphabetical order by kernel function name
    error = clCreateKernelsInProgram(
        *program,
        ALG_NAME_CAPS(NUM_KERNELS),
        kernel_buf,
        NULL
        );
    check_error(error, "Error creating kernels.");
}

void build_program(
    char *filename,
    cl_program *program,
    device *cpu,
    device *gpu
    )
{
    size_t file_size = get_file_size(filename);
    cl_char *kernel_src = (cl_char *) malloc(sizeof(cl_char) * file_size + 1); //add 1 for null terminator
    cl_uint success = read_src(
        filename,
        file_size,
        kernel_src
        );
    if (!success)
    {
        printf("Unable to read kernel source file.");
        exit(1);
    }
    
    cl_int error;
    *program = clCreateProgramWithSource(
        cpu->context, //we have the same context for both CPU and GPU
        1,
        (const char **) &kernel_src,
        NULL,
        &error
        );
    check_error(error, "Error creating program from source code.");

    cl_device_id dev_ids_buf[2];
    dev_ids_buf[0] = cpu->device_id;
    dev_ids_buf[1] = gpu->device_id;
    
    error = clBuildProgram(
        *program,
        2,
        dev_ids_buf,
        NULL, //KERNEL_BUILD_OPTS,
        NULL,
        NULL
        );
    if (error != CL_SUCCESS)
    {
        //dump build log to stdout
        printf("Build failed. Error Code=%d\n", error);

        cl_uint i;
        for (i = 0; i < 2; i++)
        {
            //get size of build log
            size_t log_size;
            clGetProgramBuildInfo(
                *program,
                dev_ids_buf[i],
                CL_PROGRAM_BUILD_LOG,
                0,
                NULL,
                &log_size
                );
            cl_char *buffer = (cl_char *) malloc(sizeof(cl_char) * log_size);

            //retreive actual build log
            clGetProgramBuildInfo(
                *program,
                dev_ids_buf[i],
                CL_PROGRAM_BUILD_LOG,
                log_size,
                buffer,
                NULL);
            
            printf("--- %s Build Log ---\n", i ? "GPU" : "CPU");
            printf("%s\n\n", buffer);

            free(buffer);
        }
        
        exit(1);
    }
    
    free(kernel_src);
}

void shutdown_opencl(
    cl_program *program,
    cl_kernel *kernels,
    device *cpu,
    device *gpu
    )
{
    cl_int error;
    cl_uint i;
    for (i = 0; i < ALG_NAME_CAPS(NUM_KERNELS); i++)
    {
        error = clReleaseKernel(kernels[i]);
        check_error(error, "Error releasing kernel object.");
    }

    clReleaseProgram(*program);
    check_error(error, "Error releasing program object.");

    error = clReleaseCommandQueue(cpu->cmd_q);
    check_error(error, "Error releasing CPU command queue.");
    
    error = clReleaseCommandQueue(gpu->cmd_q);
    check_error(error, "Error releasing GPU command queue.");

    //only do this for one device since they both share the same context
    error = clReleaseContext(gpu->context);
    check_error(error, "Error releasing context.");
}

