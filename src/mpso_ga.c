#include "mpso_ga.h"

void run_mpso_ga(
    config_ga *conf,
    cl_uint config_index,
    cl_uint num_configs,
    cl_program *program,
    cl_kernel *kernel_buf,
    device *cpu,
    device *gpu
    )
{
    device *ga_sel_dev = gpu;
    mpso_bufs_ga bufs;

    bench_fcn_info bench_info[NUM_BENCH_FCNS];
    init_bench_fcn_info(bench_info);

    profiling_data prof_data;
    init_profiling_data(
        &prof_data,
        (void *) conf,
        GA_FITNESS_SAMPLE_PTS + conf->num_swarms //space for one rep
        );

    create_mpso_bufs_ga(
        conf,
        bench_info, 
        &bufs,
        &prof_data,
        gpu
        );
    
    //start timer
    cl_uint rep;
    cl_float orig_omega = conf->omega;

    #if DEBUG
    printf("CPU id: %u\n", cpu->device_id);
    printf("GPU id: %u\n", gpu->device_id);
    printf("bench_fcn:\n");
    printf("%u\n\n", conf->bench_fcn);
    printf("m:\n");
    printf("%u\n\n", conf->m);
    printf("num_dims:\n");
    printf("%u\n\n", conf->num_dims);
    #endif

    for (rep = 0; rep < conf->num_reps; rep++)
    {
        fprintf(stderr, "Rep %u of %u\n", rep + 1, conf->num_reps);

        conf->omega = orig_omega;
        conf->max_axis_val = bench_info[conf->bench_fcn - 1].max_axis_val;
        conf->seed = get_seed();
        
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

        launch_particle_init_vec_kernel_ga(
            conf,
            kernel_buf,
            &bufs,
            gpu,
            rep
            );
        clFinish(gpu->cmd_q); //!!!!!!!!!!!!!!!
        /* clEnqueueBarrierWithWaitList( */
        /*     gpu->cmd_q, */
        /*     0, */
        /*     NULL, */
        /*     NULL */
        /*     ); */
        
        prof_data.gpu_timer.Stop();

        cl_uint i;
        cl_uint run_ga = 0;
        cl_uint prev_run_ga = run_ga;
        cl_uint last_shift_iter = 0;
        printf("%u: Running %s\n", 0, run_ga ? "GA" : "MPSO");
        for (i = 0; i < conf->max_iters; i++)
        {
            /* if (i > 0 && !(i % ALG_INTERVAL)) */
            /* { */
            /*     run_ga = !run_ga; */
            /*     //printf("%u: Running %s\n", i, run_ga ? "GA" : "MPSO"); */
            /* } */

            if (i > 0)
            {
                cl_event done;
                cl_uint *alg_health = (cl_uint *) map_buffer(
                    &(bufs.alg_health_buf),
                    CL_MAP_READ | CL_MAP_WRITE,
                    sizeof(cl_uint) * conf->num_swarms,
                    gpu,
                    &done
                    );
                clWaitForEvents(1, &done);

                cl_uint k;

                //fprintf(stderr, "%u: ", i);
                /* for (k = 0; k < conf->num_swarms; k++) */
                /* { */
                /*     fprintf(stderr, "%u ", alg_health[k]); */
                /* } */
                /* fprintf(stderr, "\n"); */
                
                for (k = 0; k < conf->num_swarms && alg_health[k] >= ALG_INTERVAL; k++);
                
                //fprintf(stderr, "i: %u, k: %u\n", i, k);

                if (k == conf->num_swarms)// || (run_ga && i - last_shift_iter >= 500))
                {
                    for (k = 0; k < conf->num_swarms; k++)
                    {
                        alg_health[k] = 0;
                    }

                    prev_run_ga = run_ga;
                    run_ga = !run_ga;
                    
                    printf("%u: Running %s\n", i, run_ga ? "GA" : "MPSO");

                    if (run_ga) //finalize fitnesses from MPSO
                    {
                        launch_update_fitness_shared_kernel_common(
                            (void *) conf,
                            kernel_buf,
                            (void *) &bufs,
                            i - last_shift_iter,
                            gpu
                            );

                        launch_update_best_vals_vec_kernel_ga(
                            conf,
                            kernel_buf,
                            &bufs,
                            i - last_shift_iter,
                            gpu
                            );
                    }
                    else //finalize fitnesses from GA
                    {
                        launch_update_fitness_shared_kernel_common(
                            (void *) conf,
                            kernel_buf,
                            (void *) &bufs,
                            i - last_shift_iter,
                            ga_sel_dev
                            );

                        launch_update_best_vals_vec_kernel_spacial(
                            conf,
                            kernel_buf,
                            &bufs,
                            i - last_shift_iter,
                            ga_sel_dev
                            );
                    }
                    
                    last_shift_iter = i;
                }

                unmap_buffer(
                    &(bufs.alg_health_buf),
                    alg_health,
                    gpu,
                    &done
                    );
                clWaitForEvents(1, &done);

                
            }

            if (run_ga)
            {
                prof_data.cpu_timer.Start();
                do_ga_alg(
                    conf,
                    &prof_data,
                    kernel_buf,
                    &bufs,
                    rep,
                    ga_sel_dev,
                    i - last_shift_iter,
                    i
                    //i == 0 || prev_run_ga != run_ga
                    );

                /* if (!((i + 1) % ALG_INTERVAL)) */
                /* { */
                /*     launch_update_fitness_shared_kernel_common( */
                /*         (void *) conf, */
                /*         kernel_buf, */
                /*         (void *) &bufs, */
                /*         i, */
                /*         ga_sel_dev */
                /*         ); */

                /*     launch_update_best_vals_vec_kernel_common( */
                /*         conf, */
                /*         kernel_buf, */
                /*         &bufs, */
                /*         i, */
                /*         ga_sel_dev */
                /*         ); */

                    /* launch_find_min_vec_kernel_common( */
                    /*     kernel_buf, */
                    /*     &(bufs.sbest_fitnesses_buf), */
                    /*     conf->num_swarms, */
                    /*     &(bufs.global_scratch_buf), */
                    /*     &(bufs.extra_data_buf), */
                    /*     rep, */
                    /*     ga_sel_dev */
                    /*     ); */
                /* } */

                if (i == conf->max_iters - 1)
                {
                    launch_update_fitness_shared_kernel_common(
                        (void *) conf,
                        kernel_buf,
                        (void *) &bufs,
                        i - last_shift_iter,
                        ga_sel_dev
                        );

                    launch_update_best_vals_vec_kernel_spacial(
                        conf,
                        kernel_buf,
                        &bufs,
                        i - last_shift_iter,
                        ga_sel_dev
                        );
                }
                
                clFinish(ga_sel_dev->cmd_q);
                /* clEnqueueBarrierWithWaitList( */
                /*     ga_sel_dev->cmd_q, */
                /*     0, */
                /*     NULL, */
                /*     NULL */
                /*     ); */
                
                prof_data.cpu_timer.Stop();
            }

            else
            {
                prof_data.gpu_timer.Start();

                do_mpso_alg(
                    conf,
                    &prof_data,
                    kernel_buf,
                    &bufs,
                    rep,
                    gpu,
                    i - last_shift_iter,
                    i
                    //i == 0 || prev_run_ga != run_ga,
                    //last_shift_iter
                    );

                /* if (!((i + 1) % ALG_INTERVAL)) */
                /* { */
                /*     launch_update_fitness_shared_kernel_common( */
                /*         (void *) conf, */
                /*         kernel_buf, */
                /*         (void *) &bufs, */
                /*         i, */
                /*         gpu */
                /*         ); */

                /*     launch_update_best_vals_vec_kernel_ga( */
                /*         conf, */
                /*         kernel_buf, */
                /*         &bufs, */
                /*         i, */
                /*         gpu */
                /*         ); */
                /* } */

                if (i == conf->max_iters - 1)
                {
                    launch_update_fitness_shared_kernel_common(
                        (void *) conf,
                        kernel_buf,
                        (void *) &bufs,
                        i - last_shift_iter,
                        gpu
                        );

                    launch_update_best_vals_vec_kernel_ga(
                        conf,
                        kernel_buf,
                        &bufs,
                        i - last_shift_iter,
                        gpu
                        );
                }
        
                clFinish(gpu->cmd_q);
                /* clEnqueueBarrierWithWaitList( */
                /*     gpu->cmd_q, */
                /*     0, */
                /*     NULL, */
                /*     NULL */
                /*     ); */
                
                prof_data.gpu_timer.Stop();
            }
        }

        //end timer
        prof_data.gpu_time[rep] = (float) prof_data.gpu_timer.GetElapsedTime();
        prof_data.cpu_time[rep] = (float) prof_data.cpu_timer.GetElapsedTime();
        prof_data.gpu_timer.Reset();
        prof_data.cpu_timer.Reset();
        
        launch_find_min_vec_kernel_common(
            kernel_buf,
            &(bufs.sbest_fitnesses_buf),
            conf->num_swarms,
            &(bufs.global_scratch_buf),
            &(bufs.final_fitness_buf),
            rep,
            gpu
            );

        clFinish(gpu->cmd_q);
        /* clEnqueueBarrierWithWaitList( */
        /*     gpu->cmd_q, */
        /*     0, */
        /*     NULL, */
        /*     NULL */
        /*     ); */
    }//end rep loop

    //this will take the final/sample fitness data from the buffer and put it into prof_data
    gather_final_fitness_data(
        &(bufs.fitness_sample_buf),
        &(bufs.final_fitness_buf),
        &(bufs.extra_data_buf),
        &prof_data,
        conf,
        gpu
        );

    release_mpso_bufs_ga(
        conf, 
        bench_info, 
        &bufs
        );

    conf->omega = orig_omega;
    print_profiling_data_ga(
        conf,
        &prof_data, 
        !config_index
        );

    release_profiling_data(
        &prof_data
        );

    clFinish(gpu->cmd_q);
}

void do_ga_alg(
    config_ga *conf,
    profiling_data *prof_data,
    cl_kernel *kernel_buf,
    mpso_bufs_ga *bufs,
    cl_uint rep,
    device *sel_dev,
    cl_uint rel_i,
    cl_uint abs_i
    )
{
    cl_uint fitness_sample_needed = 0;
    #if LAUNCH_WARNINGS
    printf("--- GA iteration %u ---\n", abs_i);
    #endif

    #if DEBUG
    print_positions(
        "Positions (CPU)",
        &(bufs->positions_buf),
        (void *) conf,
        gpu
        );
    #endif
        
    //can this use the common kernel and just launch it in a different way? But there is no local mem...
    launch_update_fitness_shared_kernel_common(
        (void *) conf,
        kernel_buf,
        (void *) bufs,
        //i % ALG_INTERVAL,
        //!recalc_sizing,
        rel_i,
        sel_dev
        );

    #if DEBUG
    print_fitnesses(
        "Fitnesses (CPU)",
        &(bufs->fitnesses_buf),
        conf,
        sel_dev
        );
    #endif

    launch_update_best_vals_vec_kernel_spacial(
        conf,
        kernel_buf,
        bufs,
        //i % ALG_INTERVAL,
        //!recalc_sizing,
        rel_i,
        sel_dev
        );

    fitness_sample_needed = !(abs_i % conf->fitness_sample_interval);
    if (fitness_sample_needed)
    {
        //prof_data->timer.Stop();
        cl_uint sample_index = rep * prof_data->num_fitness_samples + (abs_i / conf->fitness_sample_interval); //sample_index (keep in mind that this code runs only when fitness_sample_needed is true - see definition of that)
        launch_find_min_vec_kernel_common(
            kernel_buf,
            &(bufs->sbest_fitnesses_buf),
            conf->num_swarms,
            &(bufs->global_scratch_buf),
            &(bufs->fitness_sample_buf),
            sample_index,
            sel_dev
            );

        /* print_sbest_fitnesses( */
        /*     "SBest Fitnesses", */
        /*     &(bufs->sbest_fitnesses_buf), */
        /*     conf, */
        /*     sel_dev */
        /*     ); */
        /* print_n_floats( */
        /*     "Best", */
        /*     1, */
        /*     sample_index, */
        /*     &(bufs->fitness_sample_buf), */
        /*     sel_dev */
        /*     ); */
    }

    launch_cross_mut_tourn_kernel_ga(
        conf,
        kernel_buf,
        bufs,
        //i % ALG_INTERVAL,
        //!recalc_sizing,
        rel_i,
        sel_dev
        );

    /* launch_update_fitness_shared_kernel_common( */
    /*     (void *) conf, */
    /*     kernel_buf, */
    /*     (void *) bufs, */
    /*     i, */
    /*     sel_dev */
    /*     ); */

    /* launch_update_best_vals_vec_kernel_common( */
    /*     conf, */
    /*     kernel_buf, */
    /*     bufs, */
    /*     i, */
    /*     sel_dev */
    /*     ); */

    /* launch_find_min_vec_kernel_common( */
    /*     kernel_buf, */
    /*     &(bufs->sbest_fitnesses_buf), */
    /*     conf->num_swarms, */
    /*     &(bufs->global_scratch_buf), */
    /*     &(bufs->extra_data_buf), */
    /*     rep, */
    /*     sel_dev */
    /*     ); */
}

void do_mpso_alg(
    config_ga *conf,
    profiling_data *prof_data,
    cl_kernel *kernel_buf,
    mpso_bufs_ga *bufs,
    cl_uint rep,
    device *gpu,
    cl_uint rel_i,
    cl_uint abs_i
    //cl_uint recalc_sizing,
    //cl_uint last_shift_iter
    )
{
    cl_uint swap_needed = 0;
    cl_uint fitness_sample_needed = 0;
    cl_uint crossover_needed = 0;
    #if LAUNCH_WARNINGS
    printf("\n--- MPSO iteration %u ---\n", abs_i);
    #endif

    #if DEBUG
    print_positions(
        "Positions",
        &(bufs->positions_buf),
        (void *) conf,
        gpu
        );
    #endif

    launch_update_fitness_shared_kernel_common(
        (void *) conf,
        kernel_buf,
        (void *) bufs,
        //i % ALG_INTERVAL,
        //!recalc_sizing,
        rel_i,
        gpu
        );

    //if (i % ALG_INTERVAL) //there's nothing to restore on the first iteration
    if (rel_i)
    {
        launch_mut_restore_kernel_ga(
            conf,
            kernel_buf,
            bufs,
            //i % ALG_INTERVAL,
            //i - last_shift_iter == 1,
            rel_i,
            gpu
            );
    }

    #if DEBUG
    print_fitnesses(
        "Fitnesses",
        &(bufs->fitnesses_buf),
        conf,
        gpu
        );
    #endif
            
    launch_update_best_vals_vec_kernel_ga(
        conf,
        kernel_buf,
        bufs,
        //i % ALG_INTERVAL,
        //!recalc_sizing,
        rel_i,
        gpu
        );

    swap_needed = rel_i > 0 && !((rel_i + SWAP_OFFSET) % conf->exchange_iters);
    if (swap_needed)
    {
        launch_find_best_worst_alt2_kernel_common(
            (void *) conf,
            kernel_buf,
            (void *) bufs,
            //i % ALG_INTERVAL,
            //recalc_sizing ? conf->exchange_iters - 50 : i,
            rel_i,
            gpu
            );
    }

    /* print_fitnesses( */
    /*     "Fitnesses", */
    /*     &(bufs->fitnesses_buf), */
    /*     conf, */
    /*     gpu */
    /*     ); */

    #if DEBUG
    print_sbest_fitnesses(
        "SBest Fitnesses",
        &(bufs->sbest_fitnesses_buf),
        conf,
        gpu
        );
    #endif

    //fitness_sample_needed = !((i + conf->max_ga_init_iters) % conf->fitness_sample_interval);
    fitness_sample_needed = !(abs_i % conf->fitness_sample_interval);
    if (fitness_sample_needed)
    {
        //cl_uint sample_index = rep * prof_data->num_fitness_samples + ((i + conf->max_ga_init_iters) / conf->fitness_sample_interval); //sample_index (keep in mind that this code runs only when fitness_sample_needed is true - see definition of that)
        cl_uint sample_index = rep * prof_data->num_fitness_samples + (abs_i / conf->fitness_sample_interval); //sample_index (keep in mind that this code runs only when fitness_sample_needed is true - see definition of that)
        launch_find_min_vec_kernel_common(
            kernel_buf,
            &(bufs->sbest_fitnesses_buf),
            conf->num_swarms,
            &(bufs->global_scratch_buf),
            &(bufs->fitness_sample_buf),
            sample_index,
            gpu
            );

        /* print_sbest_fitnesses( */
        /*     "SBest Fitnesses", */
        /*     &(bufs->sbest_fitnesses_buf), */
        /*     conf, */
        /*     gpu */
        /*     ); */
        /* print_n_floats( */
        /*     "Best", */
        /*     1, */
        /*     sample_index, */
        /*     &(bufs->fitness_sample_buf), */
        /*     gpu */
        /*     ); */
    }

    if (rel_i && conf->omega > MIN_OMEGA) //only on second and following iterations. Doing the decay here because it ensures no update_pos_vel_kernel is in flight (kernels above block until it's done).
    {
        conf->omega *= conf->omega_decay;
    }

    crossover_needed = !(rel_i % conf->cross_iters);
    if (crossover_needed)
    {
        launch_permute_kernel_ga(
            conf,
            kernel_buf,
            bufs,
            //i % ALG_INTERVAL,
            //!recalc_sizing,
            rel_i,
            gpu
            );

        launch_crossover_kernel_ga(
            conf,
            kernel_buf,
            bufs,
            gpu
            );
    }
    else
    {
        launch_update_pos_vel_vec_kernel_ga(
            conf,
            kernel_buf,
            bufs,
            //i % ALG_INTERVAL,
            //!recalc_sizing,
            rel_i,
            rep,
            gpu
            );
    }
            
    if (swap_needed)
    {
        launch_swap_particles_vec_kernel_common(
            (void *) conf,
            kernel_buf,
            (void *) bufs,
            //i % ALG_INTERVAL,
            //recalc_sizing ? conf->exchange_iters - 50 : i,
            rel_i,
            gpu
            );
    }

    /* launch_update_fitness_shared_kernel_common( */
    /*     (void *) conf, */
    /*     kernel_buf, */
    /*     (void *) bufs, */
    /*     i, */
    /*     gpu */
    /*     ); */

    /* launch_update_best_vals_vec_kernel_ga( */
    /*     conf, */
    /*     kernel_buf, */
    /*     bufs, */
    /*     i, */
    /*     gpu */
    /*     ); */
}
