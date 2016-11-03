#include "utils.h"

//Conveniently stops the world if there's a critical error.
void check_error(
    cl_int status_cd, 
    char *patt, 
    ...
    )
{
    if (status_cd != CL_SUCCESS)
    {
        va_list arg_list;
        
        printf("\n");
        va_start(arg_list, patt);
        vprintf(patt, arg_list);
        va_end(arg_list);
        printf("\n");
        printf("Error code: %d\n", status_cd);
        printf("Error Description: %s\n", get_cl_err_cd_desc(status_cd));
        exit(1);
    }
}

char *get_kernel_name(cl_kernel *kernel)
{
    cl_int error;
    size_t len;
    char *name;

    error = clGetKernelInfo(
        *kernel,
        CL_KERNEL_FUNCTION_NAME,
        0,
        NULL,
        &len
        );
    check_error(error, "Error in get_kernel_name(), GetKernelInfo() call 1.");

    name = (char *) malloc(len * sizeof(char));

    error = clGetKernelInfo(
        *kernel,
        CL_KERNEL_FUNCTION_NAME,
        len,
        name,
        NULL
        );
    check_error(error, "Error in get_kernel_name(), GetKernelInfo() call 2.");
    
    return name;
}

char *get_cl_err_cd_desc(cl_int cd)
{
    char *desc = "unavailable";
    switch(cd)
    {
    case CL_SUCCESS:
        desc = "SUCCESS";
        break;
    case CL_DEVICE_NOT_FOUND:
        desc = "DEVICE_NOT_FOUND";
        break;
    case CL_DEVICE_NOT_AVAILABLE:
        desc = "DEVICE_NOT_AVAILABLE";
        break;
    case CL_COMPILER_NOT_AVAILABLE:
        desc = "COMPILER_NOT_AVAILABLE";
        break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        desc = "MEM_OBJECT_ALLOCATION_FAILURE";
        break;
    case CL_OUT_OF_RESOURCES:
        desc = "OUT_OF_RESOURCES";
        break;
    case CL_OUT_OF_HOST_MEMORY:
        desc = "OUT_OF_HOST_MEMORY";
        break;
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        desc = "PROFILING_INFO_NOT_AVAILABLE";
        break;
    case CL_MEM_COPY_OVERLAP:
        desc = "MEM_COPY_OVERLAP";
        break;
    case CL_IMAGE_FORMAT_MISMATCH:
        desc = "IMAGE_FORMAT_MISMATCH";
        break;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        desc = "IMAGE_FORMAT_NOT_SUPPORTED";
        break;
    case CL_BUILD_PROGRAM_FAILURE:
        desc = "BUILD_PROGRAM_FAILURE";
        break;
    case CL_MAP_FAILURE:
        desc = "MAP_FAILURE";
        break;
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        desc = "MISALIGNED_SUB_BUFFER_OFFSET";
        break;
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        desc = "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        break;
    case CL_COMPILE_PROGRAM_FAILURE:
        desc = "COMPILE_PROGRAM_FAILURE";
        break;
    case CL_LINKER_NOT_AVAILABLE:
        desc = "LINKER_NOT_AVAILABLE";
        break;
    case CL_LINK_PROGRAM_FAILURE:
        desc = "LINK_PROGRAM_FAILURE";
        break;
    case CL_DEVICE_PARTITION_FAILED:
        desc = "DEVICE_PARTITION_FAILED";
        break;
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        desc = "KERNEL_ARG_INFO_NOT_AVAILABLE";
        break;
    case CL_INVALID_VALUE:
        desc = "INVALID_VALUE";
        break;
    case CL_INVALID_DEVICE_TYPE:
        desc = "INVALID_DEVICE_TYPE";
        break;
    case CL_INVALID_PLATFORM:
        desc = "INVALID_PLATFORM";
        break;
    case CL_INVALID_DEVICE:
        desc = "INVALID_DEVICE";
        break;
    case CL_INVALID_CONTEXT:
        desc = "INVALID_CONTEXT";
        break;
    case CL_INVALID_QUEUE_PROPERTIES:
        desc = "INVALID_QUEUE_PROPERTIES";
        break;
    case CL_INVALID_COMMAND_QUEUE:
        desc = "INVALID_COMMAND_QUEUE";
        break;
    case CL_INVALID_HOST_PTR:
        desc = "INVALID_HOST_PTR";
        break;
    case CL_INVALID_MEM_OBJECT:
        desc = "INVALID_MEM_OBJECT";
        break;
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        desc = "INVALID_IMAGE_FORMAT_DESCRIPTOR";
        break;
    case CL_INVALID_IMAGE_SIZE:
        desc = "INVALID_IMAGE_SIZE";
        break;
    case CL_INVALID_SAMPLER:
        desc = "INVALID_SAMPLER";
        break;
    case CL_INVALID_BINARY:
        desc = "INVALID_BINARY";
        break;
    case CL_INVALID_BUILD_OPTIONS:
        desc = "INVALID_BUILD_OPTIONS";
        break;
    case CL_INVALID_PROGRAM:
        desc = "INVALID_PROGRAM";
        break;
    case CL_INVALID_PROGRAM_EXECUTABLE:
        desc = "INVALID_PROGRAM_EXECUTABLE";
        break;
    case CL_INVALID_KERNEL_NAME:
        desc = "INVALID_KERNEL_NAME";
        break;
    case CL_INVALID_KERNEL_DEFINITION:
        desc = "INVALID_KERNEL_DEFINITION";
        break;
    case CL_INVALID_KERNEL:
        desc = "INVALID_KERNEL";
        break;
    case CL_INVALID_ARG_INDEX:
        desc = "INVALID_ARG_INDEX";
        break;
    case CL_INVALID_ARG_VALUE:
        desc = "INVALID_ARG_VALUE";
        break;
    case CL_INVALID_ARG_SIZE:
        desc = "INVALID_ARG_SIZE";
        break;
    case CL_INVALID_KERNEL_ARGS:
        desc = "INVALID_KERNEL_ARGS";
        break;
    case CL_INVALID_WORK_DIMENSION:
        desc = "INVALID_WORK_DIMENSION";
        break;
    case CL_INVALID_WORK_GROUP_SIZE:
        desc = "INVALID_WORK_GROUP_SIZE";
        break;
    case CL_INVALID_WORK_ITEM_SIZE:
        desc = "INVALID_WORK_ITEM_SIZE";
        break;
    case CL_INVALID_GLOBAL_OFFSET:
        desc = "INVALID_GLOBAL_OFFSET";
        break;
    case CL_INVALID_EVENT_WAIT_LIST:
        desc = "INVALID_EVENT_WAIT_LIST";
        break;
    case CL_INVALID_EVENT:
        desc = "INVALID_EVENT";
        break;
    case CL_INVALID_OPERATION:
        desc = "INVALID_OPERATION";
        break;
    case CL_INVALID_GL_OBJECT:
        desc = "INVALID_GL_OBJECT";
        break;
    case CL_INVALID_BUFFER_SIZE:
        desc = "INVALID_BUFFER_SIZE";
        break;
    case CL_INVALID_MIP_LEVEL:
        desc = "INVALID_MIP_LEVEL";
        break;
    case CL_INVALID_GLOBAL_WORK_SIZE:
        desc = "INVALID_GLOBAL_WORK_SIZE";
        break;
    case CL_INVALID_PROPERTY:
        desc = "INVALID_PROPERTY";
        break;
    case CL_INVALID_IMAGE_DESCRIPTOR:
        desc = "INVALID_IMAGE_DESCRIPTOR";
        break;
    case CL_INVALID_COMPILER_OPTIONS:
        desc = "INVALID_COMPILER_OPTIONS";
        break;
    case CL_INVALID_LINKER_OPTIONS:
        desc = "INVALID_LINKER_OPTIONS";
        break;
    case CL_INVALID_DEVICE_PARTITION_COUNT:
        desc = "INVALID_DEVICE_PARTITION_COUNT";
        break;
    }

    return desc;
}

void init_profiling_data(
    profiling_data *data,
    void *generic_conf,
    cl_uint extra_data_elements //per rep
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config)*) generic_conf;
    
    data->gpu_time = (cl_float *) malloc(conf->num_reps * sizeof(cl_float));
    data->cpu_time = (cl_float *) malloc(conf->num_reps * sizeof(cl_float));

    //per rep
    //preprocessor to the rescue!
    #if ALG == ALG_GA
    //data->num_fitness_samples = (conf->max_iters + conf->max_ga_init_iters) / conf->fitness_sample_interval;
    data->num_fitness_samples = conf->max_iters / conf->fitness_sample_interval + (conf->max_iters % conf->fitness_sample_interval > 0 ? 1 : 0);
    
    #else
    data->num_fitness_samples = conf->max_iters / conf->fitness_sample_interval + (conf->max_iters % conf->fitness_sample_interval > 0 ? 1 : 0);
    #endif

    //allocate space for all reps
    data->final_fitness = (cl_float *) malloc(sizeof(cl_float) * conf->num_reps);

    //allocate space for all reps
    data->fitness_samples = (cl_float *) malloc(sizeof(cl_float) * data->num_fitness_samples * conf->num_reps);

    //per rep
    data->extra_data_elements = extra_data_elements;
    if (extra_data_elements)
    {
        //allocate space for all reps
        data->extra_data = (cl_float *) malloc(sizeof(cl_float) * extra_data_elements * conf->num_reps);
    }
}

void release_profiling_data(
    profiling_data *data
    )
{
    free(data->fitness_samples);
    free(data->final_fitness);
    free(data->gpu_time);
    free(data->cpu_time);
    
    if (data->extra_data_elements)
    {
        free(data->extra_data);
    }
}

void print_timestamp()
{
    time_t posix_time;
    time(&posix_time);
    struct tm *cur_time = localtime(&posix_time);
    
    printf("Timestamp,%s\n", asctime(cur_time));
}

void print_header_row(
    output_col *cols,
    cl_uint num_cols,
    profiling_data *prof_data,
    void *generic_conf
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config)*) generic_conf;
    
    cl_uint i;
    for (i = 0; i < num_cols; i++)
    {
        printf("%s", cols[i].header);
        if (i < num_cols - 1)
        {
            printf(",");
        }
    }

    for (i = 0; i < prof_data->num_fitness_samples; i++)
    {
        printf(",");
        printf("Sample %u", i + 1);
    }
    printf(",Avg Fitness,Rep Solve Count,Rep Fail Count,Avg Solve Flatline,Avg Fail Flatline,");

    for (i = 0; i < conf->num_reps; i++)
    {
        printf("Rep %d Flatline", i + 1);
        if (i < conf->num_reps - 1)
        {
            printf(",");
        }
    }
    
    printf("\n");
}

void print_result_row(
    output_col *cols,
    cl_uint num_cols,
    void **data_items,
    profiling_data *prof_data,
    void *generic_conf
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config)*) generic_conf;
    
    cl_uint i;
    for (i = 0; i < num_cols; i++)
    {
        switch(cols[i].data_type)
        {
        case TYPE_UINT:
            printf(cols[i].pattern, *((cl_uint *) data_items[i]));
            break;
        case TYPE_FLOAT:
            printf(cols[i].pattern, *((cl_float *) data_items[i]));
            break;
        case TYPE_STRING:
            printf(cols[i].pattern, (char *) data_items[i]);
            break;
        case TYPE_AVG:
            printf(cols[i].pattern, ((avg_val *) data_items[i])->avg, ((avg_val *) data_items[i])->std_dev);
            break;
        default:
            printf("?");
            break;
        }
        
        if (i < num_cols - 1)
        {
            printf(",");
        }
    }

    //samples
    cl_float avg_fitness;
    cl_float std_dev;
    for (i = 0; i < prof_data->num_fitness_samples; i++)
    {
        printf(",");

        get_fitness_value(
            prof_data->fitness_samples,
            prof_data->num_fitness_samples,
            i,
            conf->num_reps,
            &avg_fitness,
            &std_dev
            );

        printf("%0.10f\u00b1%0.10f", avg_fitness, std_dev);
    }

    //final fitness
    printf(",");
    get_fitness_value(
        prof_data->final_fitness,
        1,
        0,
        conf->num_reps,
        &avg_fitness,
        &std_dev
        );
    printf("%0.10f\u00b1%0.10f,", avg_fitness, std_dev);

    //iteration index data
    cl_int j;
    cl_int *rep_flatline_iter = (cl_int *) malloc(sizeof(cl_int) * conf->num_reps);
    cl_float avg_solve_flatline = 0; //avg of solved iteration index across all reps that solved the function
    cl_float avg_fail_flatline = 0; //avg of flatline iteration index across all reps that did not solve the function
    cl_int rep_solve_count = 0;
    cl_int rep_fail_count = 0;
    for (i = 0; i < conf->num_reps; i++)
    {
        for (j = prof_data->num_fitness_samples - 1; j > -1 && prof_data->fitness_samples[i * prof_data->num_fitness_samples + j] == prof_data->final_fitness[i]; j--);
        
        if ((cl_uint) j == prof_data->num_fitness_samples - 1)
        {
            rep_flatline_iter[i] = conf->max_iters;
        }
        else
        {
            rep_flatline_iter[i] = (j + 1) * conf->fitness_sample_interval;
        }

        if (prof_data->final_fitness[i] == 0)
        {
            rep_solve_count++;
            avg_solve_flatline += (float) rep_flatline_iter[i];
        }
        else
        {
            rep_fail_count++;
            avg_fail_flatline += (float) rep_flatline_iter[i];
        }
    }
    avg_solve_flatline = rep_solve_count ? avg_solve_flatline / (float) rep_solve_count : -1;
    avg_fail_flatline = rep_fail_count ? avg_fail_flatline / (float) rep_fail_count : -1;

    printf("%d,%d,%f,%f,", rep_solve_count, rep_fail_count, avg_solve_flatline, avg_fail_flatline);

    for (i = 0; i < conf->num_reps; i++)
    {
        printf("%d (%d)", rep_flatline_iter[i], prof_data->final_fitness[i] == 0);
        if (i < conf->num_reps - 1)
        {
            printf(",");
        }
    }

    /* for (i = 0; i < conf->num_reps; i++) */
    /* { */
    /*     for (j = 0; j < (cl_int) prof_data->num_fitness_samples; j++) */
    /*     { */
    /*         fprintf(stderr, "%0.10f ", prof_data->fitness_samples[i * prof_data->num_fitness_samples + j]); */
    /*     } */
    /*     fprintf(stderr, " - %d\n", rep_flatline_iter[i]); */
        
    /*     for (j = 0; j < (cl_int) prof_data->num_fitness_samples; j++) */
    /*     { */
    /*         fprintf(stderr, "%d ", prof_data->fitness_samples[i * prof_data->num_fitness_samples + j] == prof_data->final_fitness[i]); */
    /*     } */
    /*     fprintf(stderr, " - %0.10f\n", prof_data->final_fitness[i]); */
    /* } */
    
    free(rep_flatline_iter);
    printf("\n");
}

void get_fitness_value(
    cl_float *samples_buf,
    cl_uint num_samples,
    cl_uint fitness_offset,
    cl_uint num_reps,
    cl_float *avg_fitness,
    cl_float *std_dev
    )
{
    cl_uint j;

    //find avg fitness
    *avg_fitness = 0.0f; //for one sample point
    for (j = 0; j < num_reps; j++)
    {
        *avg_fitness += samples_buf[j * num_samples + fitness_offset];
    }
    *avg_fitness /= (cl_float) num_reps;

    //find std dev
    *std_dev = 0.0f;
    cl_float diff;
    for (j = 0; j < num_reps; j++)
    {
        diff = *avg_fitness - samples_buf[j * num_samples + fitness_offset];
        *std_dev += (diff * diff);
    }
    *std_dev = sqrt(*std_dev / (cl_float) num_reps);
    //printf("in fitness: %f-%f\n", *avg_fitness, *std_dev);
}

avg_val get_avg_muts(
    void *generic_conf,
    profiling_data *data,
    cl_uint extra_data_offset
    )
{
    ALG_NAME(config) *conf = (ALG_NAME(config)*) generic_conf;
    
    cl_float *avg_muts_per_rep = (cl_float *) malloc(conf->num_reps * sizeof(cl_float));
    
    cl_uint i;
    cl_uint j;
    for (i = 0; i < conf->num_reps; i++)
    {
        cl_float accum = 0;
        for (j = 0; j < conf->num_swarms; j++)
        {
            //find avg number of muts across all swarms, for each rep - store the result in the avg_muts_per_rep buffer
            accum += data->extra_data[extra_data_offset + i * conf->num_swarms + j];
        }
        avg_muts_per_rep[i] = accum / (cl_float) conf->num_swarms;
    }

    avg_val avg_muts;
    get_fitness_value(
        avg_muts_per_rep,
        1,
        0,
        conf->num_reps,
        &(avg_muts.avg),
        &(avg_muts.std_dev)
        );
    
    free(avg_muts_per_rep);

    return avg_muts;
}

void print_profiling_data_alt(
    config_alt *conf, 
    profiling_data *data, 
    cl_uint print_headers,
    char *swarm_config_str
    )
{
    const cl_uint num_cols = 17;
    output_col cols[num_cols] = {
        {"bench_fcn", TYPE_UINT, "%u"},
        {"swarm_types", TYPE_STRING, "%s"},
        {"num_swarms", TYPE_UINT, "%u"},
        {"num_sparticles", TYPE_UINT, "%u"},
        {"max_iters", TYPE_UINT, "%u"},
        {"omega", TYPE_FLOAT, "%f"},
        {"omega_decay", TYPE_FLOAT, "%f"},
        {"c1", TYPE_FLOAT, "%f"},
        {"c2", TYPE_FLOAT, "%f"},
        {"exchange_iters", TYPE_UINT, "%u"},
        {"num_exchange", TYPE_UINT, "%u"},
        {"max_axis_val", TYPE_FLOAT, "%f"},
        {"max_vel", TYPE_FLOAT, "%f"},
        {"num_dims", TYPE_UINT, "%u"},
        {"num_reps", TYPE_UINT, "%u"},
        {"m", TYPE_UINT, "%u"},
        {"Avg Execution Time", TYPE_AVG, "%f\u00b1%f"}
    };

    avg_val avg_exec_time;
    get_fitness_value(
        data->gpu_time,
        1,
        0,
        conf->num_reps,
        &(avg_exec_time.avg),
        &(avg_exec_time.std_dev)
        );

    void *data_items[num_cols] = {
        (void *) &(conf->bench_fcn),
        (void *) swarm_config_str,
        (void *) &(conf->num_swarms),
        (void *) &(conf->num_sparticles),
        (void *) &(conf->max_iters),
        (void *) &(conf->omega),
        (void *) &(conf->omega_decay),
        (void *) &(conf->c1),
        (void *) &(conf->c2),
        (void *) &(conf->exchange_iters),
        (void *) &(conf->num_exchange),
        (void *) &(conf->max_axis_val),
        (void *) &(conf->max_vel),
        (void *) &(conf->num_dims),
        (void *) &(conf->num_reps),
        (void *) &(conf->m),
        (void *) &avg_exec_time
    };

    if (print_headers)
    {
        print_header_row(
            cols,
            num_cols,
            data,
            conf
            );
    }

    print_result_row(
        cols,
        num_cols,
        data_items,
        data,
        (void *) conf
        );
}

void print_profiling_data_ga(
    config_ga *conf, 
    profiling_data *data, 
    cl_uint print_headers
    )
{
    const cl_uint num_cols = 25;
    output_col cols[num_cols] = {
        {"bench_fcn", TYPE_UINT, "%u"}, //type fitness (takes 2 elements)
        {"num_swarms", TYPE_UINT, "%u"},
        {"num_sparticles", TYPE_UINT, "%u"},
        {"max_iters", TYPE_UINT, "%u"},
        //{"max_ga_init_iters", TYPE_UINT, "%u"},
        {"omega", TYPE_FLOAT, "%f"},
        {"omega_decay", TYPE_FLOAT, "%f"},
        {"c1", TYPE_FLOAT, "%f"},
        {"c2", TYPE_FLOAT, "%f"},
        {"exchange_iters", TYPE_UINT, "%u"},
        {"num_exchange", TYPE_UINT, "%u"},
        {"max_axis_val", TYPE_FLOAT, "%f"},
        {"max_vel", TYPE_FLOAT, "%f"},
        {"num_dims", TYPE_UINT, "%u"},
        {"num_reps", TYPE_UINT, "%u"},
        {"m", TYPE_UINT, "%u"},
        {"ga_cross_ratio", TYPE_FLOAT, "%f"},
        {"ga_mut_prob", TYPE_FLOAT, "%f"},
        {"ga_tourn_size", TYPE_UINT, "%u"},
        {"cross_iters", TYPE_UINT, "%u"},
        {"unhealthy_iters", TYPE_UINT, "%u"},
        {"mut_prob", TYPE_FLOAT, "%f"},
        {"Avg muts/swarm", TYPE_AVG, "%f\u00b1%f"},
        {"Avg Execution Time (CPU)", TYPE_AVG, "%f\u00b1%f"},
        {"Avg Execution Time (GPU)", TYPE_AVG, "%f\u00b1%f"},
        {"GA End Fitness", TYPE_AVG, "%f\u00b1%f"}
    };

    avg_val avg_gpu_time;
    get_fitness_value(
        data->gpu_time,
        1,
        0,
        conf->num_reps,
        &(avg_gpu_time.avg),
        &(avg_gpu_time.std_dev)
        );

    avg_val avg_cpu_time;
    get_fitness_value(
        data->cpu_time,
        1,
        0,
        conf->num_reps,
        &(avg_cpu_time.avg),
        &(avg_cpu_time.std_dev)
        );

    avg_val avg_muts = get_avg_muts(
        (void *) conf,
        data,
        conf->num_reps //offset of num_reps because GA fitness samples are taking up that space
        );

    avg_val avg_ga_end;
    get_fitness_value(
        data->extra_data,
        1,
        0,
        conf->num_reps,
        &(avg_ga_end.avg),
        &(avg_ga_end.std_dev)
        );

    void *data_items[num_cols] = {
        (void *) &(conf->bench_fcn),
        (void *) &(conf->num_swarms),
        (void *) &(conf->num_sparticles),
        (void *) &(conf->max_iters),
        //(void *) &(conf->max_ga_init_iters),
        (void *) &(conf->omega),
        (void *) &(conf->omega_decay),
        (void *) &(conf->c1),
        (void *) &(conf->c2),
        (void *) &(conf->exchange_iters),
        (void *) &(conf->num_exchange),
        (void *) &(conf->max_axis_val),
        (void *) &(conf->max_vel),
        (void *) &(conf->num_dims),
        (void *) &(conf->num_reps),
        (void *) &(conf->m),
        (void *) &(conf->ga_cross_ratio),
        (void *) &(conf->ga_mut_prob),
        (void *) &(conf->ga_tourn_size),
        (void *) &(conf->cross_iters),
        (void *) &(conf->unhealthy_iters),
        (void *) &(conf->mut_prob),
        (void *) &avg_muts,
        (void *) &avg_cpu_time,
        (void *) &avg_gpu_time,
        (void *) &avg_ga_end
    };

    if (print_headers)
    {
        print_header_row(
            cols,
            num_cols,
            data,
            conf
            );
    }

    print_result_row(
        cols,
        num_cols,
        data_items,
        data,
        (void *) conf
        );

    /* printf("\n"); */
    /* cl_uint i; */
    /* for (i = 0; i < data->num_fitness_samples; i++) */
    /* { */
    /*     printf("%f", data->fitness_samples[i]); */
    /*     if (i < data->num_fitness_samples - 1) */
    /*     { */
    /*         printf(","); */
    /*     } */
    /* } */
    /* printf("\n"); */
}

void print_profiling_data_reg(
    config_reg *conf, 
    profiling_data *data,
    cl_uint print_headers
    )
{
    const cl_uint num_cols = 16;
    output_col cols[num_cols] = {
        {"bench_fcn", TYPE_UINT, "%u"},
        {"num_swarms", TYPE_UINT, "%u"},
        {"num_sparticles", TYPE_UINT, "%u"},
        {"max_iters", TYPE_UINT, "%u"},
        {"omega", TYPE_FLOAT, "%f"},
        {"omega_decay", TYPE_FLOAT, "%f"},
        {"c1", TYPE_FLOAT, "%f"},
        {"c2", TYPE_FLOAT, "%f"},
        {"exchange_iters", TYPE_UINT, "%u"},
        {"num_exchange", TYPE_UINT, "%u"},
        {"max_axis_val", TYPE_FLOAT, "%f"},
        {"max_vel", TYPE_FLOAT, "%f"},
        {"num_dims", TYPE_UINT, "%u"},
        {"num_reps", TYPE_UINT, "%u"},
        {"m", TYPE_UINT, "%u"},
        {"Avg Execution Time", TYPE_AVG, "%f\u00b1%f"}
    };

    avg_val avg_exec_time;
    get_fitness_value(
        data->gpu_time,
        1,
        0,
        conf->num_reps,
        &(avg_exec_time.avg),
        &(avg_exec_time.std_dev)
        );
    
    void *data_items[num_cols] = {
        (void *) &(conf->bench_fcn),
        (void *) &(conf->num_swarms),
        (void *) &(conf->num_sparticles),
        (void *) &(conf->max_iters),
        (void *) &(conf->omega),
        (void *) &(conf->omega_decay),
        (void *) &(conf->c1),
        (void *) &(conf->c2),
        (void *) &(conf->exchange_iters),
        (void *) &(conf->num_exchange),
        (void *) &(conf->max_axis_val),
        (void *) &(conf->max_vel),
        (void *) &(conf->num_dims),
        (void *) &(conf->num_reps),
        (void *) &(conf->m),
        (void *) &avg_exec_time
    };

    if (print_headers)
    {
        print_header_row(
            cols,
            num_cols,
            data,
            conf
            );
    }

    print_result_row(
        cols,
        num_cols,
        data_items,
        data,
        (void *) conf
        );
}

void print_profiling_data_mcs(
    config_mcs *conf, 
    profiling_data *data, 
    cl_uint print_headers
    )
{
    //allocate num_swarms extra space for the mut_counts
    const cl_uint num_cols = 21;

    output_col cols[num_cols] = {
        {"bench_fcn", TYPE_UINT, "%u"},
        {"num_swarms", TYPE_UINT, "%u"},
        {"num_sparticles", TYPE_UINT, "%u"},
        {"max_iters", TYPE_UINT, "%u"},
        {"omega", TYPE_FLOAT, "%f"},
        {"omega_decay", TYPE_FLOAT, "%f"},
        {"c1", TYPE_FLOAT, "%f"},
        {"c2", TYPE_FLOAT, "%f"},
        {"exchange_iters", TYPE_UINT, "%u"},
        {"num_exchange", TYPE_UINT, "%u"},
        {"max_axis_val", TYPE_FLOAT, "%f"},
        {"max_vel", TYPE_FLOAT, "%f"},
        {"num_dims", TYPE_UINT, "%u"},
        {"num_reps", TYPE_UINT, "%u"},
        {"m", TYPE_UINT, "%u"},
        {"cross_iters", TYPE_UINT, "%u"},
        {"unhealthy_ratio", TYPE_FLOAT, "%f"},
        {"unhealthy_iters", TYPE_UINT, "%u"},
        {"mut_prob", TYPE_FLOAT, "%f"},
        {"Avg muts/swarm", TYPE_AVG, "%f\u00b1%f"},
        {"Avg Execution Time", TYPE_AVG, "%f\u00b1%f"}
    };

    avg_val avg_muts = get_avg_muts(
        (void *) conf,
        data,
        0
        );

    avg_val avg_exec_time;
    get_fitness_value(
        data->gpu_time,
        1,
        0,
        conf->num_reps,
        &(avg_exec_time.avg),
        &(avg_exec_time.std_dev)
        );
    
    void *data_items[num_cols] = {
        (void *) &(conf->bench_fcn),
        (void *) &(conf->num_swarms),
        (void *) &(conf->num_sparticles),
        (void *) &(conf->max_iters),
        (void *) &(conf->omega),
        (void *) &(conf->omega_decay),
        (void *) &(conf->c1),
        (void *) &(conf->c2),
        (void *) &(conf->exchange_iters),
        (void *) &(conf->num_exchange),
        (void *) &(conf->max_axis_val),
        (void *) &(conf->max_vel),
        (void *) &(conf->num_dims),
        (void *) &(conf->num_reps),
        (void *) &(conf->m),
        (void *) &(conf->cross_iters),
        (void *) &(conf->unhealthy_ratio),
        (void *) &(conf->unhealthy_iters),
        (void *) &(conf->mut_prob),
        (void *) &avg_muts,
        (void *) &avg_exec_time
    };

    if (print_headers)
    {
        print_header_row(
            cols,
            num_cols,
            data,
            conf
            );
    }

    print_result_row(
        cols,
        num_cols,
        data_items,
        data,
        (void *) conf
        );
}

void print_profiling_data_tm(
    config_tm *conf, 
    profiling_data *data, 
    cl_uint print_headers
    )
{
    const cl_uint num_cols = 13;
    output_col cols[num_cols] = {
        {"num_swarms", TYPE_UINT, "%u"},
        {"num_sparticles", TYPE_UINT, "%u"},
        {"max_iters", TYPE_UINT, "%u"},
        {"omega", TYPE_FLOAT, "%f"},
        {"omega_decay", TYPE_FLOAT, "%f"},
        {"c1", TYPE_FLOAT, "%f"},
        {"c2", TYPE_FLOAT, "%f"},
        {"exchange_iters", TYPE_UINT, "%u"},
        {"num_exchange", TYPE_UINT, "%u"},
        {"max_axis_val", TYPE_UINT, "%u"},
        {"num_dims", TYPE_UINT, "%u"},
        {"num_reps", TYPE_UINT, "%u"},
        {"Avg Execution Time", TYPE_AVG, "%f\u00b1%f"}
    };

    avg_val avg_exec_time;
    get_fitness_value(
        data->gpu_time,
        1,
        0,
        conf->num_reps,
        &(avg_exec_time.avg),
        &(avg_exec_time.std_dev)
        );
    
    void *data_items[num_cols] = {
        (void *) &(conf->num_swarms),
        (void *) &(conf->num_sparticles),
        (void *) &(conf->max_iters),
        (void *) &(conf->omega),
        (void *) &(conf->omega_decay),
        (void *) &(conf->c1),
        (void *) &(conf->c2),
        (void *) &(conf->exchange_iters),
        (void *) &(conf->num_exchange),
        (void *) &(conf->max_axis_val),
        (void *) &(conf->num_dims),
        (void *) &(conf->num_reps),
        (void *) &avg_exec_time
    };

    if (print_headers)
    {
        print_header_row(
            cols,
            num_cols,
            data,
            conf
            );
    }

    print_result_row(
        cols,
        num_cols,
        data_items,
        data,
        (void *) conf
        );
}

//generates random real values in the range [-max_val, max_val]
void fill_rand_buf(
    cl_float *buf, 
    cl_uint n, 
    cl_float max_val
    )
{
    cl_uint i;
    unsigned int val;
    for (i = 0; i < n; i++)
    {
        #if FIXED_SEED
        val = FIXED_SEED;
        #else
        rand_s(&val);
        #endif
        //expand the value to [0, 2*max_val], then subtract max_val to make range [-max_val, max_val]
        buf[i] = u01_closed_closed_32_24(val) * max_val * 2 - max_val;
    }
}

//generates random uint values in the range [min_val, max_val]
void fill_rand_uint_buf(
    cl_uint *buf,
    cl_uint n,
    cl_uint min_val,
    cl_uint max_val
    )
{
    cl_uint i;
    unsigned int val;
    for (i = 0; i < n; i++)
    {
        #if FIXED_SEED
        val = i;
        #else
        rand_s(&val);
        #endif

        buf[i] = (val + min_val) % max_val;
    }
}

cl_uint get_seed()
{
    cl_uint seed;
    #if FIXED_SEED
    seed = FIXED_SEED;
    #else
    rand_s(&seed);
    #endif

    return seed;
}

//rounds up
//ensures that n is a multiple of m and n >= m
cl_uint make_multiple(
    cl_uint n, 
    cl_uint m
    )
{
    if (n < m)
    {
        //next multiple of m is m itself
        n = m;
    }
    else if (n > m)
    {
        //round up to next multiple of m
        n = (n / m) * m + ((n % m > 0) ? m : 0);
    }

    return n;
}

size_t get_file_size(
    char *filename
    )
{
    cl_long file_len = 0L;
    
    FILE *file = fopen(filename, "r");
    if (file)
    {
        fseek(file, 0L, SEEK_END);
        file_len = ftell(file);
        fclose(file);
    }

    return file_len > 0 ? (size_t) file_len : (size_t) 0;
}

//Reads and returns (kernel) source code from a given file and puts it into src.
cl_uint read_src(
    char *filename,
    size_t file_size,
    cl_char *src)
{
    cl_uint success = 0;
    FILE *file = fopen(filename, "r");

    success = file != NULL;
    if (success)
    {
        success = ( fread(src, sizeof(cl_char), file_size, file) == (size_t) file_size );
        if (success)
        {
            //make sure last null terminator is in place
            src[file_size] = '\0';
        }
        
        fclose(file);
    }

    return success;
}

void fill_perm_buf(
    cl_uint *buf,
    cl_uint n
    )
{
    #if FIXED_SEED
    cl_uint i;
    for (i = 0; i < n; i++)
    {
        #if PERMUTE
        buf[i] = i;
        #else
        buf[i] = n - i - 1;
        #endif
    }

    #else
    unsigned int rand_val;
    cl_uint i;
    cl_uint j;
    cl_int temp;

    //fill the buffer
    for (i = 0; i < n; i++)
    {
        buf[i] = i;
    }

    #if PERMUTE
    //perform "modern" Fisher-Yates shuffle
    for (i = n - 1; i > 0; i--)
    {
        rand_s(&rand_val);
        
        j = rand_val % (i + 1); //0 <= j <= i

        //swap buf[j] and buf[i]
        temp = buf[j];
        buf[j] = buf[i];
        buf[i] = temp;
    }
    #endif
    #endif
}

cl_float *gen_orthogonal_matrix(
    cl_float *matrix, //mxm matrix
    cl_uint dim_len
    )
{
    int i, j, k;
    double dp, t;
    unsigned int rand_val;

loop:
    for (;;)
    {
        for (i = (dim_len - 1); i >= 0; i--)
        {
            for (j = (dim_len - 1); j >= 0; j--)
            {
                #if FIXED_SEED
                rand_val = i * (dim_len - 1) + j;
                #else
                rand_s(&rand_val);
                #endif
                
                matrix[i * dim_len + j] = u01_closed_closed_32_24(rand_val);
            }
        }

        //main loop of gram/schmidt
        for (i = (dim_len - 1); i >= 0; i--)
        {
            for (j = (dim_len - 1); j > i; j--)
            {
                //dot product
                dp = 0;

                for (k = (dim_len - 1); k >= 0; k--)
                {
                    dp += (matrix[i * dim_len + k] * matrix[j * dim_len + k]);
                }

                //subtract
                for (k = (dim_len - 1); k >= 0; k--)
                {
                    matrix[i * dim_len + k] -= (dp * matrix[j * dim_len + k]);
                }
            }

            //normalize
            dp = 0;

            for (k = (dim_len - 1); k >= 0; k--)
            {
                t   = matrix[i * dim_len + k];
                dp += (t * t);
            }

            //linear dependency -> restart
            if (dp <= 0)
            {
                goto loop;
            }

            dp = (1 / sqrt(dp));

            for (k = (dim_len - 1); k >= 0; k--)
            {
                matrix[i * dim_len + k] *= dp;
            }
        }

        return matrix;
    }
}

cl_float *gen_identity_matrix(
    cl_float *matrix, //m x m matrix
    cl_uint dim_len
    )
{
    cl_uint i;
    cl_uint j;
    
    for (i = 0; i < dim_len; i++)
    {
        for (j = 0; j < dim_len; j++)
        {
            matrix[i * dim_len + j] = (cl_float) (i == j);
        }
    }

    return matrix;
}

//this code adapted from lsgo_benchmark sample (CEC 2012 website)
cl_float *fill_rot_matrix_buf(
    cl_float *matrix, //m x m matrix
    cl_uint dim_len
    )
{
    #if ROTATE
    return gen_orthogonal_matrix(
        matrix,
        dim_len
        );
    #else
    return gen_identity_matrix(
        matrix,
        dim_len
        );
    #endif
}
