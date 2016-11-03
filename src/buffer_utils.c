#include "buffer_utils.h"

void *map_buffer(
    cl_mem *buf,
    cl_map_flags map_flags,
    size_t size,
    device *gpu,
    cl_event *complete
    )
{
    cl_int error;
    void *host_buf = (void *) clEnqueueMapBuffer(
        gpu->cmd_q,
        *buf,
        CL_FALSE,
        map_flags,
        0,
        size,
        0,
        NULL,
        complete,
        &error
        );
    check_error(error, "Error mapping buffer.");

    return host_buf;
}

void unmap_buffer(
    cl_mem *buf,
    void *mapped_ptr,
    device *gpu,
    cl_event *complete
    )
{
    cl_int error = clEnqueueUnmapMemObject(
        gpu->cmd_q,
        *buf,
        mapped_ptr,
        0,
        NULL,
        complete
        );
    check_error(error, "Error unmapping buffer.");
}

void fill_rot_matrix_buf(
    cl_mem *buf,
    cl_uint rot_matrix_dim_len,
    device *gpu
    )
{
    cl_event event;
    cl_float *host_buf = (cl_float *) map_buffer(
        buf,
        CL_MAP_WRITE_INVALIDATE_REGION,
        rot_matrix_dim_len * rot_matrix_dim_len * sizeof(cl_float),
        gpu,
        &event
        );
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    
    fill_rot_matrix_buf(
        host_buf,
        rot_matrix_dim_len
        );

    unmap_buffer(
        buf,
        host_buf,
        gpu,
        &event
        );
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
}

void fill_perm_buf(
    cl_mem *buf,
    void *generic_conf,
    device *gpu
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) (generic_conf);
    cl_event event;
    cl_uint *host_buf = (cl_uint *) map_buffer(
        buf,
        CL_MAP_WRITE_INVALIDATE_REGION,
        conf->num_dims * sizeof(cl_uint),
        gpu,
        &event
        );
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    
    fill_perm_buf(
        host_buf,
        conf->num_dims
        );

    unmap_buffer(
        buf,
        host_buf,
        gpu,
        &event
        );
    clWaitForEvents(1, &event);

    clReleaseEvent(event);
}

void fill_optimum_buf(
    cl_mem *buf,
    void *generic_conf,
    device *gpu
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config) *) (generic_conf);
    cl_event event;
    cl_float *host_buf = (cl_float *) map_buffer(
        buf,
        CL_MAP_WRITE_INVALIDATE_REGION,
        conf->num_dims * sizeof(cl_float),
        gpu,
        &event
        );
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
        
    #if FIXED_SEED
    cl_uint i;
    for (i = 0; i < conf->num_dims; i++)
    {
        host_buf[i] = -50;
    }
    #else
    /* cl_uint i; */
    /* for (i = 0; i < conf->num_dims; i++) */
    /* { */
    /*     host_buf[i] = 0; */
    /* } */
    fill_rand_buf(
        host_buf,
        conf->num_dims,
        conf->max_axis_val
        );
    #endif

    unmap_buffer(
        buf,
        host_buf,
        gpu,
        &event
        );
    clWaitForEvents(1, &event);

    clReleaseEvent(event);
}

void gather_final_fitness_data(
    cl_mem *fitness_sample_buf,
    cl_mem *final_fitness_buf,
    cl_mem *extra_data_buf,
    profiling_data *prof_data,
    void *generic_conf,
    device *gpu
    )
{
    #if LAUNCH_WARNINGS
    printf("Gathering fitness data...\n");
    #endif
    
    ALG_NAME(config) *conf = (ALG_NAME(config) *) (generic_conf);
    
    //note: here we use clEnqueueReadBuffer to copy back to the host, rather than clEnqueueMapBuffer.
    //The reason for this is that readBuffer avoids the slow USWC lines (a copy is faster than reading across them here, since these buffers were allocated using CL_MEM_USE_PERSISTENT_MEM_AMD).

    cl_uint num_events = prof_data->extra_data_elements ? 3 : 2;
    cl_event *read_events = (cl_event *) malloc(sizeof(cl_event) * num_events);
    cl_int error;
    cl_uint i;

    cl_float *host_sample_buf;
    cl_float *host_final_fitness_buf;
    cl_float *host_extra_data_buf;

    host_sample_buf = (cl_float *) malloc(prof_data->num_fitness_samples * conf->num_reps * sizeof(cl_float));
    host_final_fitness_buf = (cl_float *) malloc(conf->num_reps * sizeof(cl_float));

    error = clEnqueueReadBuffer(
        gpu->cmd_q,
        *fitness_sample_buf,
        CL_FALSE,
        0,
        prof_data->num_fitness_samples * conf->num_reps * sizeof(cl_float),
        host_sample_buf,
        0,
        NULL,
        &(read_events[0])
        );
    check_error(error, "Error reading sample buf to host.");

    error = clEnqueueReadBuffer(
        gpu->cmd_q,
        *final_fitness_buf,
        CL_FALSE,
        0,
        sizeof(cl_float) * conf->num_reps,
        host_final_fitness_buf,
        0,
        NULL,
        &(read_events[1])
        );
    check_error(error, "Error reading final fitness buf to host.");

    //copy extra data to host (if necessary)
    if (prof_data->extra_data_elements)
    {
        host_extra_data_buf = (cl_float *) malloc(prof_data->extra_data_elements * sizeof(cl_float) * conf->num_reps);
        error = clEnqueueReadBuffer(
            gpu->cmd_q,
            *extra_data_buf,
            CL_FALSE,
            0,
            prof_data->extra_data_elements * conf->num_reps * sizeof(cl_float),
            host_extra_data_buf,
            0,
            NULL,
            &(read_events[2])
            );
        check_error(error, "Error reading extra data buf to host.");
    }
    //barrier
    clWaitForEvents(num_events, read_events);

    //copy the read data to appropriate places
    for (i = 0; i < prof_data->num_fitness_samples * conf->num_reps; i++)
    {
        prof_data->fitness_samples[i] = host_sample_buf[i];
    }

    for (i = 0; i < conf->num_reps; i++)
    {
        prof_data->final_fitness[i] = host_final_fitness_buf[i];
    }

    for (i = 0; i < prof_data->extra_data_elements * conf->num_reps; i++)
    {
        prof_data->extra_data[i] = host_extra_data_buf[i];
    }

    //clean up
    for (i = 0; i < num_events; i++)
    {
        clReleaseEvent(read_events[i]);
    }
    free(read_events);
    free(host_sample_buf);
    free(host_final_fitness_buf);
    if (prof_data->extra_data_elements)
    {
        free(host_extra_data_buf);
    }

    #if LAUNCH_WARNINGS
    printf("Done gathering fitness data.\n");
    #endif
}
