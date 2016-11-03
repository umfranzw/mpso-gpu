__kernel void update_pos_vel_vec(
    __global float *positions,               //size is num_swarms * num_sparticles * num_tasks
    __global float *velocities,              //size is num_swarms * num_sparticles * num_tasks
    __global float *particle_best_positions, //size is num_swarms * num_sparticles * num_tasks
    //__constant float *swarm_best_positions,    //size is num_swarms * num_tasks
    __global float *swarm_best_positions,    //size is num_swarms * num_tasks
    __global uint *swarm_health,
    __global float *fitnesses,
    __global float *pre_mut_positions,
    __global float *pre_mut_velocities,
    __global float *pre_mut_fitnesses,
    __global float *mut_counts,
    uint num_swarms,
    uint num_sparticles,
    uint num_dims,
    float omega,
    float c1,
    float c2,
    float max_axis_val,
    uint seed,
    uint iter_index,
    uint unhealthy_iters,
    float mut_prob,
    float max_vel,
    uint rep,
    uint num_reps
    )
{
    uint global_id = get_global_id(0);
    uint swarm_id = global_id / (num_sparticles * num_dims / 4);
    uint particle_id = (global_id / (num_dims / 4)) % num_sparticles;
    uint dim_id = global_id % (num_dims / 4);
    uint health = swarm_health[swarm_id];

    float4 pos_chunk = vload4(0, positions + global_id * 4);
    float4 vel_chunk = vload4(0, velocities + global_id * 4);
    float4 new_vel_chunk;
    float4 new_pos_chunk;

    float4 r1 = get_float_rands_vec(
        global_id,
        UPDATE_POS_VEL_STREAM,
        iter_index * 2,
        seed
        );
    float4 r2 = get_float_rands_vec(
        global_id,
        UPDATE_POS_VEL_STREAM,
        iter_index * 2 + 1,
        seed
        );
    
    if (mut_prob <= 0.0f || health < unhealthy_iters)
    {
        float4 pbest_chunk = vload4(0, particle_best_positions + global_id * 4);
        float4 sbest_chunk = vload4(0, swarm_best_positions + swarm_id * num_dims + dim_id * 4);

        new_vel_chunk = omega * vel_chunk +
            c1 * r1 *
            (pbest_chunk - pos_chunk) +
            c2 * r2 *
            (sbest_chunk - pos_chunk);

        new_pos_chunk = pos_chunk + new_vel_chunk;
    }

    //mutation or pso update, depending on a random number comparison
    else if (health >= unhealthy_iters)
    {
        float4 mut_r = get_float_rands_vec(
            global_id,
            MUTATION_STREAM,
            iter_index,
            seed
            );

        float4 pbest_chunk = vload4(0, particle_best_positions + global_id * 4);
        float4 sbest_chunk = vload4(0, swarm_best_positions + swarm_id * num_dims + dim_id * 4);

        float4 pso_vel_chunk = omega * vel_chunk +
            c1 * r1 *
            (pbest_chunk - pos_chunk) +
            c2 * r2 *
            (sbest_chunk - pos_chunk);

        float4 pso_pos_chunk = pos_chunk + pso_vel_chunk;

        //note: we can use the same r1 and r2 here, since each dimension will be selected for one of mutation or standard pso update (below)
        float4 mut_pos_chunk = pos_chunk + ( r1 - (float4) (0.5f) ) * pos_chunk;
        float4 mut_vel_chunk = vel_chunk + ( r2 - (float4) (0.5f) ) * vel_chunk;

        int4 cmp = (mut_r < mut_prob);
        new_pos_chunk = select(pso_pos_chunk, mut_pos_chunk, cmp);
        new_vel_chunk = select(pso_vel_chunk, mut_vel_chunk, cmp);

        //store the original (unmutated) position, velocity, and fitness for potential restoration on the next iteration
        vstore4(pos_chunk, 0, pre_mut_positions + global_id * 4);
        vstore4(vel_chunk, 0, pre_mut_velocities + global_id * 4);
        
        uint fit_particle_id = global_id - (swarm_id * num_sparticles * num_dims / 4);
        
        if (fit_particle_id < num_sparticles / 4)
        {
            float4 old_fitness = vload4(0, fitnesses + swarm_id * num_sparticles + fit_particle_id * 4);
            
            vstore4(old_fitness, 0, pre_mut_fitnesses + swarm_id * num_sparticles + fit_particle_id * 4);
        }

        if (!particle_id && !dim_id)
        {
            mut_counts[rep * num_swarms + swarm_id + num_reps] += 1; //leave first num_reps elements for GA fitness samples
        }
    }

    //clamp new values to the solution space
    new_pos_chunk = clamp( new_pos_chunk, (float4) -1.0f * max_axis_val, (float4) max_axis_val );
    new_vel_chunk = clamp( new_vel_chunk, (float4) (-1.0f * max_vel), (float4) (max_vel) );

    //store new positions and velocities
    vstore4(new_vel_chunk, 0, velocities + global_id * 4);
    vstore4(new_pos_chunk, 0, positions + global_id * 4);
}
