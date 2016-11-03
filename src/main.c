#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include "clhelper.h"
#include "config_utils.h"
#include ALG_HEADER_STR(mpso)
#include ALG_HEADER_STR(kernels)

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: mpso-gpu <conf_file>\n");
        exit(1);
    }

    FILE *config_file = fopen(
        //CONFIG_FILE_STR(),
        argv[1],
        "r"
        );

    device cpu;
    device gpu;
    cl_program program;
    cl_kernel kernels[ALG_NAME_CAPS(NUM_KERNELS)];

    init_opencl(
        KERNELS_FILENAME,
        &program,
        kernels,
        &cpu,
        &gpu
        );

    print_timestamp();

    ALG_NAME(config) conf;
    cl_uint num_configs = get_num_configs(config_file);
    cl_int have_config = ALG_NAME(get_next_config)(
        config_file, 
        &conf
        );
    cl_uint config_index = 0;
    while (have_config)
    {
        fprintf(stderr, "Running \"%s\": config %u of %u\n", argv[1], config_index + 1, num_configs);
        
        ALG_NAME(run_mpso)(
            &conf,
            config_index,
            num_configs,
            &program,
            kernels,
            &cpu,
            &gpu
            //&cpu
            );

        config_index++;
        have_config = ALG_NAME(get_next_config)(
            config_file, 
            &conf
            );
    }
    
    shutdown_opencl(
        &program, 
        (cl_kernel *) kernels,
        &cpu,
        &gpu
        );

    fclose(config_file);

    return 0;
}
