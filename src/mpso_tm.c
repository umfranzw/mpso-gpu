#include "mpso_tm.h"

void run_mpso_tm(
    config_tm *conf,
    cl_uint config_index,
    cl_uint num_configs,
    cl_program *program,
    cl_kernel *kernel_buf,
    device *cpu,
    device *gpu
    )
{
    cl_uint combined = 0;

    mpso_bufs_tm bufs;
        
    profiling_data prof_data;
    init_profiling_data(
        &prof_data,
        (void *) conf,
        0
        );

    create_mpso_bufs_tm(
        conf,
        &bufs,
        &prof_data,
        gpu
        );

    cl_uint rep;
    cl_uint i;
    cl_float orig_omega = conf->omega;
    
    for (rep = 0; rep < conf->num_reps; rep++)
    {
        fprintf(stderr, "Rep %u of %u\n", rep + 1, conf->num_reps);
        
        conf->omega = orig_omega;
        conf->max_axis_val = conf->num_machines - 1;
        conf->seed = get_seed();

        prof_data.gpu_timer.Start();

        gen_etc_matrix(
            conf,
            &bufs,
            gpu
            );

        //we can use the common driver, but the actual included kernel (in kernels_tm.cl) is tm-specific (for tm-specific range of positions)
        launch_particle_init_vec_kernel_common(
            (void *) conf,
            kernel_buf,
            (void *) &bufs,
            gpu
            );
        
        cl_uint swap_needed = 0;
        cl_uint fitness_sample_needed = 0;
        for (i = 0; i < conf->max_iters; i++)
        {
            launch_update_fitness_shared_vec_kernel_tm(
                conf,
                kernel_buf,
                &bufs,
                i,
                swap_needed,
                gpu,
                combined
                );

            launch_update_best_vals_vec_kernel_common(
                (void *) conf,
                kernel_buf,
                (void *) &bufs,
                i,
                gpu
                );

            swap_needed = !((i + 1) % conf->exchange_iters);
            if (swap_needed)
            {
                launch_find_best_worst_vec2_kernel_common(
                    conf,
                    kernel_buf,
                    &bufs,
                    i,
                    gpu
                    );
            }

            fitness_sample_needed = !(i % conf->fitness_sample_interval);
            if (fitness_sample_needed)
            {
                cl_uint sample_index = rep * (i / conf->fitness_sample_interval); //sample_index (keep in mind that this code runs only when fitness_sample_needed is true - see definition of that)
                launch_find_min_vec_kernel_common(
                    kernel_buf,
                    &(bufs.sbest_fitnesses_buf),
                    conf->num_swarms,
                    &(bufs.global_scratch_buf),
                    &(bufs.fitness_sample_buf),
                    sample_index,
                    gpu
                    );

                /* launch_update_samples_vec_kernel_common( */
                /*     kernel_buf, */
                /*     &(bufs.fitness_sample_buf), */
                /*     &(bufs.global_scratch_buf), */
                /*     prof_data.num_fitness_samples, */
                /*     i / conf->fitness_sample_interval, //sample_index (keep in mind that this code runs only when fitness_sample_needed is true - see definition of that) */
                /*     rep == 0, //init_output */
                /*     rep == (conf->num_reps - 1) ? (cl_float) conf->num_reps : 0, //divisor */
                /*     gpu */
                /*     ); */
            }

            if (i && conf->omega > MIN_OMEGA) //only on second and following iterations. Doing the decay here because it ensures no update_pos_vel_kernel is in flight (kernels above block until it's done).
            {
                conf->omega *= conf->omega_decay;
            }

            launch_update_pos_vel_vec_kernel_common(
                (void *) conf,
                kernel_buf,
                (void *) &bufs,
                i,
                gpu
                );
            
            if (swap_needed)
            {
                //this is a tm-specific driver that uses the common kernel (driver needed for the 'combined' parameter)
                launch_swap_particles_vec_kernel_tm(
                    conf,
                    kernel_buf,
                    &bufs,
                    i,
                    gpu,
                    combined
                    );
            }
        }
        clFinish(gpu->cmd_q);
    
        prof_data.gpu_timer.Stop();
        prof_data.gpu_time[rep] = (float) prof_data.gpu_timer.GetElapsedTime();
        launch_find_min_vec_kernel_common(
            kernel_buf,
            &(bufs.sbest_fitnesses_buf),
            conf->num_swarms,
            &(bufs.global_scratch_buf),
            &(bufs.final_fitness_buf),
            rep,
            gpu
            );

        /* launch_update_samples_vec_kernel_common( */
        /*     kernel_buf, */
        /*     &(bufs.final_fitness_buf), */
        /*     &(bufs.global_scratch_buf), */
        /*     1, */
        /*     0, */
        /*     rep == 0, */
        /*     rep == conf->num_reps - 1 ? conf->num_reps : 0, */
        /*     gpu */
        /*     ); */
        
        prof_data.gpu_timer.Reset();
    }
    //this will take the final/sample fitness data from the buffer and put it into prof_data
    gather_final_fitness_data(
        &(bufs.fitness_sample_buf),
        &(bufs.final_fitness_buf),
        NULL,
        &prof_data,
        conf,
        gpu
        );

    release_mpso_bufs_tm(
        &bufs
        );

    conf->omega = orig_omega;
    print_profiling_data_tm(
        conf,
        &prof_data,
        !config_index
        );

    release_profiling_data(
        &prof_data
        );
}

void gen_etc_matrix(
    config_tm *conf,
    mpso_bufs_tm *bufs,
    device *gpu
    )
{
    cl_event map_complete;
    cl_float *host_etc_buf = (cl_float *) map_buffer(
        &(bufs->etc_buf),
        CL_MAP_WRITE_INVALIDATE_REGION,
        conf->num_machines * conf->num_dims * sizeof(cl_float),
        gpu,
        &map_complete
        );
    clWaitForEvents(1, &map_complete);
    clReleaseEvent(map_complete);

    cl_uint *host_machines_buf = (cl_uint *) malloc(conf->num_machines * sizeof(cl_uint));
    fill_rand_uint_buf(
        host_machines_buf,
        conf->num_machines,
        MIN_MIPS_TM,
        MAX_MIPS_TM
        );

    cl_uint *host_tasks_buf = (cl_uint *) malloc(conf->num_dims * sizeof(cl_uint));
    fill_rand_uint_buf(
        host_tasks_buf,
        conf->num_dims,
        MIN_TASK_INST_TM,
        MAX_TASK_INST_TM
        );
    
    cl_int error = clWaitForEvents(1, &map_complete);
    clReleaseEvent(map_complete);
    check_error(error, "Error waiting for map_complete event in gen_etc_matrix().");

    cl_uint i, j;
    for (i = 0; i < conf->num_machines; i++)
    {
        for (j = 0; j < conf->num_dims; j++)
        {
            host_etc_buf[i * conf->num_dims + j] = (cl_float) host_tasks_buf[j] / (cl_float) host_machines_buf[i];
        }
    }

    printf("Machines:\n");
    print_uint_buf(
        host_machines_buf,
        conf->num_machines,
        0
        );
    
    printf("Tasks:\n");
    print_uint_buf(
        host_tasks_buf,
        conf->num_dims,
        0
        );

    printf("ETC Matrix:\n");
    print_float_matrix(
        host_etc_buf,
        conf->num_dims,
        conf->num_machines,
        1
        );

    unmap_buffer(
        &(bufs->etc_buf),
        host_etc_buf,
        gpu,
        &map_complete
        );
    clWaitForEvents(1, &map_complete);
    clReleaseEvent(map_complete);

    free(host_machines_buf);
    free(host_tasks_buf);
}
