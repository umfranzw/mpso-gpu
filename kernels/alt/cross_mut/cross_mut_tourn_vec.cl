//threads per particle: max(tourn_size / 4, num_dims / 4)
//scratch per particle: tourn_size / 2

void __kernel cross_mut_tourn_vec(
    __global float *positions, //size = num_swarms * num_sparticles * num_dims
    __global float *fitnesses,
    __constant uint *swarm_types,
    const uint num_swarms,
    const uint num_sparticles,
    const uint num_dims,
    const float cross_ratio,
    const float mut_prob,
    const uint tourn_size,
    const uint threads_per_particle,
    const uint iter_index,
    const uint seed,
    __local float* scratch
    )
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint particles_per_group = get_local_size(0) / threads_per_particle;
    uint swarm_id = global_id / (num_sparticles * threads_per_particle);
    uint particle_id = (global_id / threads_per_particle) % num_sparticles;
    uint dim_id = global_id % threads_per_particle;
    //uint particle_offset = ((swarm_id * particle_id) % particles_per_group);
    uint particle_offset = (swarm_id * num_sparticles + particle_id) % particles_per_group; //particle offset within group
    uint scratch_particle_offset = particle_offset * (tourn_size / 2);
    uint is_ga_swarm = swarm_types[swarm_id] == TYPE_GA;
    uint tourn_mask = is_ga_swarm && dim_id < (tourn_size / 4);
    uint dim_mask = is_ga_swarm && dim_id < (num_dims / 4);

    //Tournament selection: we begin by selecting our pool of individuals for the tournament, then find the one with the best (lowest) fitness.
    //No need to rearrange fitnesses, just move positions.
    uint4 t_indices;
    float4 t_vals;
    float2 left;
    float2 right;
    float min_fitness;
    if (tourn_mask)
    {
        //this is absolute particle index (across swarms)
        t_indices = get_uint_rands_vec(
            global_id,
            CROSSOVER_STREAM,
            iter_index,
            seed
            );

        t_indices = t_indices % num_sparticles;
        //t_indices += (uint4) (swarm_id * num_sparticles);

        //this is painful, but it must be done - retrieve 4 values in sequence from global memory
        t_vals = (float4) (fitnesses[swarm_id * num_sparticles + t_indices.x],
                           fitnesses[swarm_id * num_sparticles + t_indices.y],
                           fitnesses[swarm_id * num_sparticles + t_indices.z],
                           fitnesses[swarm_id * num_sparticles + t_indices.w]
            );

        /* if (!swarm_id) */
        /* { */
        /*     vstore4(convert_float4(t_indices), 0, test_tourn_indices + swarm_id * num_sparticles * tourn_size + particle_id * tourn_size + dim_id * 4); */
        /*     vstore4(t_vals, 0, test_tourn_vals + swarm_id * num_sparticles * tourn_size + particle_id * tourn_size + dim_id * 4); */
        /* } */
    
        left = fmin(t_vals.xy, t_vals.zw);
        vstore2(left, 0, scratch + scratch_particle_offset + dim_id * 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //reduction using (t / 4) / 2 threads per particle
    uint i;
    for (i = tourn_size / 8; i > 0; i /= 2)
    {
        if (tourn_mask && dim_id < i)
        {
            right = vload2(0, scratch + scratch_particle_offset + (dim_id + i) * 2);
            left = fmin(left, right);
            vstore2(left, 0, scratch + scratch_particle_offset + dim_id * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tourn_mask && !dim_id && i > 1 && i % 2)
        {
            right = vload2(0, scratch + scratch_particle_offset + (i - 1) * 2);
            left = fmin(left, right);
            vstore2(left, 0, scratch + scratch_particle_offset); //+ dim_id * 2 is redundant, since dim_id == 0 here
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //add in the odd vector, if any
    if (tourn_mask && (tourn_size / 4) > 1 && (tourn_size / 4) % 2 && !dim_id)
    {
        right = vload2(0, scratch + scratch_particle_offset + ((tourn_size / 4) - 1) * 2);
        left = min(left, right);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //perform final component-wise reduction
    if (tourn_mask && !dim_id)
    {
        min_fitness = fmin(left.x, left.y);
        scratch[scratch_particle_offset] = min_fitness;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //find the particle index of the minimum fitness
    if (tourn_mask)
    {
        min_fitness = scratch[scratch_particle_offset];

        /* if (!swarm_id) */
        /*     test_tourn_mins[swarm_id * num_sparticles + particle_id] = min_fitness; */
        
        int4 cmp = ((float4) min_fitness == t_vals);
        
        if (any(cmp)) //THIS IS NOT EVER TRUE!
        {
            //set comp to absolute particle index where we have -1, and to a sentinal value where we have 0
            t_indices = select((uint4) (num_swarms * num_sparticles), t_indices, cmp);
            //could improve this with commented findmin alg above?
            t_indices.xy = min(t_indices.xy, t_indices.zw);
            t_indices.x = min(t_indices.x, t_indices.y);

            //write to different bank to avoid need for barrier (note this this is ok because tourn_size >= 4)
            scratch[scratch_particle_offset + 1] = (float) global_id;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tourn_mask && global_id == (uint) scratch[scratch_particle_offset + 1])
    {
        scratch[scratch_particle_offset] = (float) t_indices.x;
        /* if (!swarm_id) */
        /*     test_local_mem[particle_id] = (float) t_indices.x; */
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //everybody reads the index (only right threads need it - optimize this later)
    uint sel_index;
    if (dim_mask)
    {
        sel_index = (uint) scratch[scratch_particle_offset];
        /* if (!swarm_id) */
        /*     test_tourn_sel[swarm_id * num_sparticles * threads_per_particle + particle_id * threads_per_particle + dim_id] = sel_index; */
        //right threads can load up sel index straightaway, right here (optimize later)
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //4 * threads_per_particle threads will generate the same 4-element rand vector for prob_cross - this way we can avoid using local memory, since we don't necessarily have enough (one element per particle)
    float4 rands = get_float_rands_vec(
        //swarm_id * num_sparticles + particle_id * (num_dims / 4) + dim_id,
        swarm_id * num_sparticles + particle_id,
        CROSSOVER_STREAM,
        iter_index * 2,
        seed
        );
    uint4 shift = (uint4) ((swarm_id * num_sparticles + particle_id) % 4);
    uint4 shuffle_mask = (shift + (uint4) (0, 3, 2, 1)) % (uint4) 4;
    rands = shuffle(rands, shuffle_mask);
    //now this thread's random number is at rands.x !

    uint cross_pt;
    float4 pos_chunk;
    uint crossing = rands.x < cross_ratio;
    uint cross_index;
    uint intersecting;
    if (dim_mask && crossing)
    {
        //4 * threads_per_particle threads will generate the same 4-element rand vector for cross_pt - this way we can avoid using local memory, since we don't necessarily have enough (one element per particle)
        uint4 rands = get_uint_rands_vec(
            //swarm_id * num_sparticles + particle_id * (num_dims / 4) + dim_id,
            swarm_id * num_sparticles + particle_id,
            CROSSOVER_STREAM,
            iter_index * 2 + 1,
            seed
            ) % (num_dims + 1);
        uint4 shift = (uint4) ((swarm_id * num_sparticles + particle_id) % 4);
        uint4 shuffle_mask = (shift + (uint4) (0, 3, 2, 1)) % (uint4) 4;
        //uint4 shuffle_mask = ((uint4) dim_id + (uint4) (0, 1, 2, 3)) % (uint4) 4;
        rands = shuffle(rands, shuffle_mask);
        //now this thread's random number is at rands.x !
        cross_pt = rands.x;
        //cross_pt = 16;
        
        //test2_buf[swarm_id * num_sparticles * threads_per_particle + particle_id * threads_per_particle + dim_id] = cross_pt;

        cross_index = cross_pt / 4;
        intersecting = cross_pt % 4;

        //cross pos_chunk with sel_pos_chunk, and store result in sel_pos_chunk
        if (dim_id < cross_index) //left takes pos_chunk
        {
            pos_chunk = vload4(0, positions + swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4);
        }

        else //right takes sel_pos_chunk
        {
            pos_chunk = vload4(0, positions + swarm_id * num_sparticles * num_dims + sel_index * num_dims + dim_id * 4);
            //pos_chunk = (float4) 0;
        }
    }

    /* else if (dim_mask && !crossing) */
    /* { */
    /*     test2_buf[swarm_id * num_sparticles * threads_per_particle + particle_id * threads_per_particle + dim_id] = -1; */
    /* } */

    //splice together the intersection vector, if necessary
    if (dim_mask && crossing && intersecting && cross_index == dim_id)
    {
        //float4 intersect_chunk = vload4(0, positions + sel_index * num_dims + dim_id * 4);
        float4 intersect_chunk = vload4(0, positions + swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4);
        int4 cmp = (uint4) (0, 1, 2, 3) < (uint4) intersecting;
        pos_chunk = select(pos_chunk, intersect_chunk, cmp);
    }

    //if there's no crossover, put the selected particle through
    if (dim_mask && !crossing)
    {
        pos_chunk = vload4(0, positions + sel_index * num_dims + dim_id * 4);
    }

    //Mutation
    if (dim_mask)
    {
        float4 rands_vec = get_float_rands_vec(
            global_id,
            MUTATION_STREAM,
            iter_index * 2,
            seed
            );

        int4 cmp = rands_vec < (float4) (mut_prob);
        //uniform mutation for now
        rands_vec = get_float_rands_vec(
            global_id,
            MUTATION_STREAM,
            iter_index * 2 + 1,
            seed
            );
        
        rands_vec = rands_vec * (float4) 2 - (float4) 1;
        rands_vec = select((float4) 0, rands_vec, cmp);
        pos_chunk += rands_vec;

        //pos_chunk = pos_chunk + (rands_vec - (float) (0.5f)) * pos_chunk;
    }

    //write result to global mem
    if (dim_mask)
    {
        vstore4(pos_chunk, 0, positions + swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4);        
    }    
}
