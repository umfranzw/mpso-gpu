#include "pretty_printer.h"

void print_float_matrix(
    cl_float *buf, 
    cl_uint width, 
    cl_uint height, 
    cl_uint factor
    )
{
    cl_uint i, j;
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            printf("%10f ", buf[i * width + j] * factor);
        }
        printf("\n");
    }
    printf("\n");
}

void print_uint_buf(
    cl_uint *buf, 
    cl_uint n, 
    cl_uint show_indices
    )
{
    cl_uint i;
    for (i = 0; i < n; i++)
    {
        if (show_indices)
        {
            printf("%u: ", i);
        }
        printf("%u", buf[i]);
        if (i < n - 1)
        {
            printf(", ");
        }
    }
    printf("\n\n");
}

void print_swarm_health_buf(
    char *label,
    cl_mem *mem_buf,
    void *generic_conf,
    device *gpu
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;
    
    printf("%s:\n", label);

    cl_event done;
    cl_uint *host_buf = (cl_uint *) map_buffer(
        mem_buf,
        CL_MAP_READ,
        conf->num_swarms * sizeof(cl_uint),
        gpu,
        &done
        );
    clWaitForEvents(1, &done);

    cl_uint i;
    for (i = 0; i < conf->num_swarms; i++)
    {
        printf("{%u}", host_buf[i]);
        if (i < conf->num_swarms - 1)
        {
            printf(",");
        }
        printf("\n");
    }

    unmap_buffer(
        mem_buf,
        host_buf,
        gpu,
        &done
        );
    clWaitForEvents(1, &done);
}

void print_crossover_perm_buf(
    char *label, 
    config_mcs *conf, 
    mpso_bufs_mcs *bufs, 
    device *gpu
    )
{
    cl_event done;
    cl_uint *host_buf = (cl_uint *) map_buffer(&(bufs->crossover_perm_buf), CL_MAP_READ, conf->num_swarms * conf->num_sparticles * sizeof(cl_uint), gpu, &done);
    clWaitForEvents(1, &done);
    clReleaseEvent(done);

    printf("%s:\n", label);

    cl_uint swarm;
    cl_uint particle;
    for (swarm = 0; swarm < conf->num_swarms; swarm++)
    {
        printf("{");
        for (particle = 0; particle < conf->num_sparticles; particle++)
        {
            printf("%u", host_buf[swarm * conf->num_sparticles + particle]);
            if (particle < conf->num_sparticles - 1)
            {
                printf(", ");
            }
        }
        printf("}");
        if (swarm < conf->num_swarms - 1)
        {
            printf(",\n");
        }
    }
    printf("\n");

    unmap_buffer(&(bufs->crossover_perm_buf), host_buf, gpu, &done);
    clWaitForEvents(1, &done);
    clReleaseEvent(done);
}

void print_float4(
    cl_float4 data
    )
{
    printf("(%f, %f, %f, %f)", data.s[0], data.s[1], data.s[2], data.s[3]);
}

void print_uint4(
    cl_uint4 data
    )
{
    printf("(%u, %u, %u, %u)", data.s[0], data.s[1], data.s[2], data.s[3]);
}

void print_unvec_positions(
    cl_float *buf, 
    void *generic_conf
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    cl_uint swarm, particle, dim;
    for (swarm = 0; swarm < conf->num_swarms; swarm++)
    {
        printf("{");
        for (particle = 0; particle < conf->num_sparticles; particle++)
        {
            printf("(");
            for (dim = 0; dim < conf->num_dims; dim++)
            {
                cl_float *buf_ptr = buf + swarm * conf->num_sparticles * conf->num_dims + particle * conf->num_dims + dim;
                printf("%f", *(buf_ptr));
                if (dim < conf->num_dims - 1)
                {
                    printf(", ");
                }
            }
            printf(")");
            if (particle < conf->num_sparticles - 1)
            {
                printf(", ");
            }
            printf("\n");
        }
        printf("}");
        if (swarm < conf->num_swarms - 1)
        {
            printf(",\n");
        }
    }
    printf("\n\n");
}

void print_unvec_velocities(
    cl_float *buf,
    void *generic_conf
    )
{
    print_unvec_positions(
        buf,
        generic_conf
        );
}

void print_fitnesses(
    cl_float *buf, 
    void *generic_conf
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    cl_uint swarm, particle;
    for (swarm = 0; swarm < conf->num_swarms; swarm++)
    {
        printf("{");
        for (particle = 0; particle < conf->num_sparticles; particle++)
        {
            printf("%f", buf[swarm * conf->num_sparticles + particle]);
            if (particle < conf->num_sparticles - 1)
            {
                printf(", ");
            }
        }
        printf("}");
        if (swarm < conf->num_swarms - 1)
        {
            printf(",\n");
        }
    }
    printf("\n\n");
}

void print_unvec_sbest_positions(
    cl_float *buf,
    void *generic_conf
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    cl_uint swarm, dim;
    for (swarm = 0; swarm < conf->num_swarms; swarm++)
    {
        printf("{");
        for (dim = 0; dim < conf->num_dims; dim++)
        {
            printf("%f", buf[swarm * conf->num_dims + dim]);
            if (dim < conf->num_dims - 1)
            {
                printf(", ");
            }
        }
        printf("}");
        if (swarm < conf->num_swarms - 1)
        {
            printf(",\n");
        }
    }
    printf("\n\n");
}

void print_sbest_fitnesses(
    cl_float *buf, 
    void *generic_conf
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    cl_uint swarm;
    for (swarm = 0; swarm < conf->num_swarms; swarm++)
    {
        printf("{%f}", buf[swarm]);
        
        if (swarm < conf->num_swarms - 1)
        {
            printf(",\n");
        }
    }
    printf("\n\n");
}

void print_indices_buf(
    cl_uint *buf, 
    void *generic_conf
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    cl_uint swarm;
    cl_uint index;
    for (swarm = 0; swarm < conf->num_swarms; swarm++)
    {
        printf("{");
        for (index = 0; index < conf->num_exchange; index++)
        {
            printf("%u", buf[swarm * conf->num_exchange + index]);

            if (index < conf->num_exchange - 1)
            {
                printf(",");
            }
        }
        printf("}");

        if (swarm < conf->num_swarms - 1)
        {
            printf(",");
        }
        printf("\n\n");
    }
}

void print_n_floats(
    char *label, 
    cl_uint n,
    cl_uint offset,
    cl_mem *mem_buf,
    device *gpu
    )
{
    printf("%s:\n", label);
    
    cl_event done;
    cl_int error;
    cl_float *buf = (cl_float *) clEnqueueMapBuffer(
        gpu->cmd_q,
        *mem_buf,
        CL_FALSE,
        CL_MAP_READ,
        offset * sizeof(cl_float),
        n * sizeof(cl_float),
        0,
        NULL,
        &done,
        &error
        );
    check_error(error, "Error mapping buffer in print_n_floats().");
    
    //cl_float *buf = (cl_float *) map_buffer(mem_buf, CL_MAP_READ, n * sizeof(cl_float), gpu, &done);
    clWaitForEvents(1, &done);

    cl_uint i;
    for (i = 0; i < n; i++)
    {
        printf("%f", buf[i]);
        if (i < n - 1)
        {
            printf(", ");
        }
    }
    printf("\n\n");
    
    unmap_buffer(mem_buf, buf, gpu, &done);
    clWaitForEvents(1, &done);
}

void print_n_uints(
    char *label, 
    cl_uint n,
    cl_mem *mem_buf,
    device *gpu
    )
{
    printf("%s:\n", label);
    
    cl_event done;
    cl_float *buf = (cl_float *) map_buffer(mem_buf, CL_MAP_READ, n * sizeof(cl_uint), gpu, &done);
    clWaitForEvents(1, &done);

    cl_uint i;
    for (i = 0; i < n; i++)
    {
        printf("%u", buf[i]);
        if (i < n - 1)
        {
            printf(", ");
        }
        printf("\n");
    }
    printf("\n\n");
    
    unmap_buffer(mem_buf, buf, gpu, &done);
    clWaitForEvents(1, &done);
}

void print_perm_vec(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    printf("%s:\n", label);
    cl_event done;
    cl_uint *buf = (cl_uint *) map_buffer(mem_buf, CL_MAP_READ, conf->num_dims * sizeof(cl_uint), gpu, &done);
    clWaitForEvents(1, &done);

    cl_uint i;
    for (i = 0; i < conf->num_dims; i++)
    {
        printf("%u", buf[i]);
        if (i < conf->num_dims - 1)
        {
            printf(", ");
        }
    }
    printf("\n\n");

    unmap_buffer(mem_buf, buf, gpu, &done);
    clWaitForEvents(1, &done);
}

void print_rot_matrix(
    char *label,
    cl_mem *mem_buf, 
    cl_uint dim_len, 
    device *gpu
    )
{
    printf("%s:\n", label);
    cl_event done;
    cl_float *buf = (cl_float *) map_buffer(mem_buf, CL_MAP_READ, dim_len * dim_len * sizeof(cl_float), gpu, &done);
    clWaitForEvents(1, &done);

    print_float_matrix(buf, dim_len, dim_len, 1);

    unmap_buffer(mem_buf, buf, gpu, &done);
    clWaitForEvents(1, &done);
}

void print_positions(
    char *label,
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    printf("%s:\n", label);
    cl_event done;
    cl_float *buf = (cl_float *) map_buffer(mem_buf, CL_MAP_READ, conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float), gpu, &done);
    clWaitForEvents(1, &done);

    print_unvec_positions(buf, conf);

    unmap_buffer(mem_buf, buf, gpu, &done);
    clWaitForEvents(1, &done);
}

void print_velocities(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    printf("%s:\n", label);
    cl_event done;
    cl_float *buf = (cl_float *) map_buffer(mem_buf, CL_MAP_READ, conf->num_swarms * conf->num_sparticles * conf->num_dims * sizeof(cl_float), gpu, &done);
    clWaitForEvents(1, &done);

    print_unvec_velocities(buf, conf);

    unmap_buffer(mem_buf, buf, gpu, &done);
    clWaitForEvents(1, &done);
}

void print_fitnesses(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    printf("%s:\n", label);
    cl_event done;
    cl_float *buf = (cl_float *) map_buffer(mem_buf, CL_MAP_READ, conf->num_swarms * conf->num_sparticles * sizeof(cl_float), gpu, &done);
    clWaitForEvents(1, &done);

    print_fitnesses(buf, conf);

    unmap_buffer(mem_buf, buf, gpu, &done);
    clWaitForEvents(1, &done);
}

void print_sbest_positions(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    printf("%s:\n", label);
    cl_event done;
    cl_float *buf = (cl_float *) map_buffer(mem_buf, CL_MAP_READ, conf->num_swarms * conf->num_dims * sizeof(cl_float), gpu, &done);
    clWaitForEvents(1, &done);

    print_unvec_sbest_positions(buf, conf);

    unmap_buffer(mem_buf, buf, gpu, &done);
    clWaitForEvents(1, &done);
}

void print_sbest_fitnesses(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    printf("%s:\n", label);
    cl_event done;
    cl_float *buf = (cl_float *) map_buffer(mem_buf, CL_MAP_READ, conf->num_swarms * sizeof(cl_float), gpu, &done);
    clWaitForEvents(1, &done);

    print_sbest_fitnesses(buf, conf);

    unmap_buffer(mem_buf, buf, gpu, &done);
    clWaitForEvents(1, &done);
}

void print_indices_buf(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) generic_conf;

    printf("%s:\n", label);
    cl_event done;
    cl_uint *buf = (cl_uint *) map_buffer(mem_buf, CL_MAP_READ, conf->num_swarms * conf->num_exchange * sizeof(cl_uint), gpu, &done);
    clWaitForEvents(1, &done);

    print_indices_buf(buf, conf);

    unmap_buffer(mem_buf, buf, gpu, &done);
    clWaitForEvents(1, &done);
}
