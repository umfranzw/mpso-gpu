//mapping: one thread per 4 dims, straight through workgroups (no local memory needed) - i.e. threads operating on a particle may be split between multiple workgroups

void __kernel cross_mut_pbest_vec(
    __global float *positions, //size = num_swarms * num_sparticles * num_dims
    __global float *pbest_positions,
    __global uint *crossover_perm, //size = num_swarms * num_sparticles
    __constant uint *swarm_types,
    const uint num_swarms,
    const uint num_sparticles,
    const uint num_dims,
    const float max_axis_val,
    const float cross_ratio,
    const float mut_prob,
    const uint iter_index,
    const uint seed
    )
{
    uint global_id = get_global_id(0);
    uint abs_particle_id = global_id / (num_dims / 4);
    uint swarm_id = abs_particle_id / num_sparticles;
    uint dim_id = global_id % (num_dims / 4);
    //uint particle_mask = abs_particle_id < num_swarms * num_sparticles; //for last workgroup, which may not need all the threads
        
    if (swarm_types[swarm_id] == TYPE_GA)// && particle_mask)
    {
        //crossover

        //all threads operating on the same particle will generate the same 4-element rand vector for prob_cross - this way we can avoid using local memory for communication of the rands
        /* float4 rands = get_float_rands_vec( */
        /*     abs_particle_id, */
        /*     CROSSOVER_STREAM, */
        /*     iter_index, */
        /*     seed */
        /*     ); */
    
        /* uint cross_particle_offset = all_swarms_cross ? */
        /*     ((swarm_id + 1) % num_swarms) * num_sparticles * num_dims + crossover_perm[abs_particle_id] + dim_id * 4 : */
        /*     swarm_id * num_sparticles * num_dims + crossover_perm[abs_particle_id] + dim_id * 4; */
        float4 new_pos;
        
        float4 rands = get_float_rands_vec(
            global_id / 4, //every 4 threads generate the same vector
            CROSSOVER_STREAM,
            iter_index,
            seed
            );
        //pick out this threads random value from the vector of 4
        uint4 mask = (uint4) (global_id % 4, (global_id + 1) % 4, (global_id + 2) % 4, (global_id + 3) % 4);
        rands = shuffle(rands, mask); //now this thread's random number is in position x
        float r = rands.x;

        if (r < cross_ratio)
        {
            uint cross_particle_offset = swarm_id * num_sparticles * num_dims + crossover_perm[abs_particle_id] + dim_id * 4;

            float4 cur_pbest = vload4(0, pbest_positions + global_id * 4);
            float4 rand_pbest = vload4(0, pbest_positions + cross_particle_offset);

            new_pos = (cur_pbest + rand_pbest) / (float4) (2.0f);
        }
        else
        {
            //no crossover, load current position in preparation for mutation below
            new_pos = vload4(0, positions + global_id * 4);
        }

        //mutation

        rands = get_float_rands_vec(
            global_id,
            MUTATION_STREAM,
            iter_index,
            seed
            );

        float4 mut_pos = new_pos + ( rands - (float4) (0.5f) ) * new_pos;
        int4 cmp = rands < (float4) (mut_prob);
        new_pos = select(new_pos, mut_pos, cmp);

        //clamp and store
        new_pos = clamp( new_pos, (float4) -1.0f * max_axis_val, (float4) max_axis_val );
        vstore4(new_pos, 0, positions + global_id * 4);
    }
}
