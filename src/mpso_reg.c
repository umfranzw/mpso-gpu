#include "mpso_reg.h"

void run_mpso_reg(
    config_reg *conf,
    cl_uint config_index,
    cl_uint num_configs,
    cl_program *program,
    cl_kernel *kernel_buf,
    device *cpu,
    device *gpu
    )
{
    mpso_bufs_reg bufs;

    bench_fcn_info bench_info[NUM_BENCH_FCNS];
    init_bench_fcn_info(bench_info);
        
    profiling_data prof_data;
    init_profiling_data(
        &prof_data,
        (void *) conf,
        0
        );

    create_mpso_bufs_reg(
        conf, 
        bench_info, 
        &bufs,
        &prof_data,
        gpu
        );

    //start timer
    cl_uint rep;
    cl_uint i;
    cl_float orig_omega = conf->omega;

    for (rep = 0; rep < conf->num_reps; rep++)
    {
        fprintf(stderr, "Rep %u of %u\n", rep + 1, conf->num_reps);

        conf->omega = orig_omega;
        conf->max_axis_val = bench_info[conf->bench_fcn - 1].max_axis_val;
        conf->seed = get_seed();

        #if DEBUG
        printf("num_dims:\n");
        printf("%u\n\n", conf->num_dims);
        printf("m:\n");
        printf("%u\n\n", conf->m);
        printf("bench_fcn:\n");
        printf("%u\n\n", conf->bench_fcn);
        #endif

        if (bench_info[conf->bench_fcn - 1].need_opt_vec)
        {
            fill_optimum_buf(
                &(bufs.optimum_buf),
                (void *) conf,
                gpu
                );

            #if DEBUG
            print_n_floats(
                "Optimum",
                conf->num_dims,
                0,
                &(bufs.optimum_buf),
                gpu
                );
            #endif
        }

        if (bench_info[conf->bench_fcn - 1].need_rot_matrix)
        {
            fill_rot_matrix_buf(
                &(bufs.initial_rot_matrix_buf),
                conf->m,
                gpu
                );

            #if DEBUG
            print_rot_matrix(
                "Initial Rotation Matrix",
                &(bufs.initial_rot_matrix_buf),
                conf->m,
                gpu
                );
            #endif

            launch_init_rot_matrix_kernel_common(
                (void *) conf,
                kernel_buf,
                (void *) &bufs,
                gpu
                );

            /* #if DEBUG */
            /* print_rot_matrix( */
            /*     "Transposed Rotation Matrix", */
            /*     &(bufs.rot_matrix_buf), */
            /*     conf->m, */
            /*     gpu */
            /*     ); */
            /* #endif */
        }

        if (bench_info[conf->bench_fcn - 1].need_perm_vec)
        {
            fill_perm_buf(
                &(bufs.perm_vec_buf),
                (void *) conf,
                gpu
                );

            #if DEBUG
            print_perm_vec(
                "Permutation Vector",
                &(bufs.perm_vec_buf),
                (void *) conf,
                gpu
                );
            #endif
        }
        clFinish(gpu->cmd_q);
        /* clEnqueueBarrierWithWaitList( */
        /*     gpu->cmd_q, */
        /*     0, */
        /*     NULL, */
        /*     NULL */
        /*     ); */

        prof_data.gpu_timer.Start();

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
            #if LAUNCH_WARNINGS
            printf("--- Iteration %u ---\n", i + 1);
            #endif

            #if DEBUG
            print_positions(
                "Positions",
                &(bufs.positions_buf),
                (void *) conf,
                gpu
                );
            #endif

            launch_update_fitness_shared_kernel_common(
                (void *) conf,
                kernel_buf,
                (void *) &bufs,
                i,
                gpu
                );

            #if DEBUG
            print_fitnesses(
                "Fitnesses",
                &(bufs.fitnesses_buf),
                (void *) conf,
                gpu
                );
            #endif

            launch_update_best_vals_vec_kernel_common(
                (void *) conf,
                kernel_buf,
                (void *) &bufs,
                i,
                gpu
                );

            /* print_sbest_fitnesses( */
            /*     "SBest Fitnesses", */
            /*     &(bufs.sbest_fitnesses_buf), */
            /*     conf, */
            /*     gpu */
            /*     ); */

            swap_needed = i > 0 && !(i % conf->exchange_iters);
            if (swap_needed)
            {
                launch_find_best_worst_alt2_kernel_common(
                    (void *) conf,
                    kernel_buf,
                    (void *) &bufs,
                    i,
                    gpu
                    );
            }

            fitness_sample_needed = !(i % conf->fitness_sample_interval);
            if (fitness_sample_needed)
            {
                //prof_data.timer.Stop();
                //sample_index (keep in mind that this code runs only when fitness_sample_needed is true - see definition of that)
                cl_uint sample_index = i / conf->fitness_sample_interval;
                launch_find_min_vec_kernel_common(
                    kernel_buf,
                    &(bufs.sbest_fitnesses_buf),
                    conf->num_swarms,
                    &(bufs.global_scratch_buf),
                    &(bufs.fitness_sample_buf),
                    rep * prof_data.num_fitness_samples + sample_index,
                    gpu
                    );

                /* launch_update_samples_vec_kernel_common( */
                /*     kernel_buf, */
                /*     &(bufs.fitness_sample_buf), */
                /*     &(bufs.global_scratch_buf), */
                /*     prof_data.num_fitness_samples, */
                /*     sample_index,  */
                /*     rep == 0, //init_output */
                /*     (rep == conf->num_reps - 1) && (sample_index == prof_data.num_fitness_samples - 1) ? (cl_float) conf->num_reps : 0, //divisor */
                /*     gpu */
                /*     ); */
                
                //prof_data.timer.Start();
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
                launch_swap_particles_vec_kernel_common(
                    (void *) conf,
                    kernel_buf,
                    (void *) &bufs,
                    i,
                    gpu
                    );
            }
        }//end iter loop

        launch_update_fitness_shared_kernel_common(
            (void *) conf,
            kernel_buf,
            (void *) &bufs,
            i,
            gpu
            );

        launch_update_best_vals_vec_kernel_common(
            (void *) conf,
            kernel_buf,
            (void *) &bufs,
            i,
            gpu
            );

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
        /*     (rep == conf->num_reps - 1) ? (cl_float) conf->num_reps : 0, */
        /*     gpu */
        /*     ); */

        clFinish(gpu->cmd_q);
        /* clEnqueueBarrierWithWaitList( */
        /*     gpu->cmd_q, */
        /*     0, */
        /*     NULL, */
        /*     NULL */
        /*     ); */
    
        //end timer
        prof_data.gpu_timer.Stop();
        prof_data.gpu_time[rep] = (float) prof_data.gpu_timer.GetElapsedTime();
        prof_data.gpu_timer.Reset();
    }//end rep loop

    //this will take the final/sample fitness data from the buffer and put it into prof_data
    gather_final_fitness_data(
        &(bufs.fitness_sample_buf),
        &(bufs.final_fitness_buf),
        NULL,
        &prof_data,
        conf,
        gpu
        );

    release_mpso_bufs_reg(
        conf, 
        bench_info, 
        &bufs
        );

    conf->omega = orig_omega;
    print_profiling_data_reg(
        conf,
        &prof_data, 
        !config_index
        );

    release_profiling_data(
        &prof_data
        );

    clFinish(gpu->cmd_q);
}
