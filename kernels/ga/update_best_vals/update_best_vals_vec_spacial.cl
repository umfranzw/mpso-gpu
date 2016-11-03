//num_swarms workgroups of 256 (num_sparticles) threads each
__kernel void update_best_vals_vec(
    __global float *fitness, //size = num_swarms * num_sparticles
    __global float *position, //size = num_swarms * num_sparticles * num_dims
    __global float *pbest_fitness,   //particle best fitnesses (size = num_swarms * num_sparticles)
    __global float *pbest_position,  //particle best positions (size = num_swarms * num_sparticles * num_dims)
    __global float *sbest_fitness,   //swarm best fitnesses (size = num_swarms)
    __global float *sbest_position,  //swarm best positions (size = num_swarms * num_dims)
    __local float *fitness_scratch, //local memory scratch space (size = num_sparticles)
    uint num_swarms,
    uint num_sparticles,
    uint num_dims,
    float unhealthy_ratio,
    __global uint *alg_health
    )
{
    uint swarms_per_group = get_local_size(0) / num_sparticles;
    uint group_id = get_group_id(0) * swarms_per_group + get_local_id(0) / num_sparticles; //each id corresponds to one swarm (there are num_swarms workgroups in total)
    uint local_id = get_local_id(0) % num_sparticles; //each id corresponds to an offset within a swarm (each workgroup contains num_sparticles threads)

    uint position_offset = group_id * num_sparticles * num_dims + local_id * num_dims;

    float4 update_occurred = (float4) (-1.0f); //< 0 means no update occurred
    float2 partial = (float2) (0.0f);

    uint group_improved = 0;

    //a subset of threads update the pbest fitnesses
    if (local_id < num_sparticles / 4)// && group_id < num_swarms)
    {
        float4 fitness_chunk = vload4(0, fitness + group_id * num_sparticles + local_id * 4);
        float4 pbest_chunk = vload4(0, pbest_fitness + group_id * num_sparticles + local_id * 4);
        int4 cmp = (fitness_chunk < pbest_chunk) || (pbest_chunk == (float4) (FLT_MAX)); //on the first iteration, pbest_chunk will be equal to FLT_MAX. In the event the fitness_chunk also happens to be equal to FLT_MAX (which is possible, though extremely unlikely), we want to perform the update even though fitness_chunk == pbest_chunk.
        float4 new_fitnesses = select(pbest_chunk, fitness_chunk, cmp);

        if (any(cmp))
        {
            //store the updated values in pbest_fitnesses
            vstore4( new_fitnesses, 0, pbest_fitness + group_id * num_sparticles + local_id * 4 );
        }

        //update fitness_scratch - store a -1 if no update occurred, and the new fitness value if an update did occur
        update_occurred = select(update_occurred, new_fitnesses, cmp);
        vstore4( update_occurred, 0, fitness_scratch + (group_id % swarms_per_group) * num_sparticles + local_id * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float per_thread_fitness = fitness_scratch[get_local_id(0)];
    float4 pos_vec;
    uint i;
    if (local_id < num_sparticles * swarms_per_group)// && group_id < num_swarms)
    {
        //all threads update the pbest positions
        if (per_thread_fitness >= 0)
        {
            for (i = 0; i < num_dims / 4; i++)
            {
                pos_vec = vload4(0, position + position_offset + i * 4);
                vstore4(pos_vec, 0, pbest_position + position_offset + i * 4);
            }
        }
    }

    //perform parallel reduction to find swarm best fitness (use float2s so we can support a swarm size of 4 (using float4s means we need at least 8 particles/swarm for the reduction to work)
    float2 left;
    float2 right;
    float2 updated_vec;
    float2 zero = (float2) (0.0f, 0.0f);
    float2 max_vec = (float2) (FLT_MAX, FLT_MAX);
    uint group_swarm_index = group_id % swarms_per_group;
    uint swarm_base = group_swarm_index * num_sparticles;

    for (i = num_sparticles / 4; i > 0; i /= 2) //start at num_sparticles / 4 because we divide by 2, and then each thread handles 2 elements at a time => 2 * 2 = 4 = 2^2
    {
        if (local_id < i)// && group_id < num_swarms)
        {
            left = vload2(0, fitness_scratch + swarm_base + local_id * 2);
            right = vload2(0, fitness_scratch + swarm_base + (local_id + i) * 2);
            left = select(left, max_vec, left < zero);
            right = select(right, max_vec, right < zero);
            updated_vec = fmin(left, right);
            vstore2(
                updated_vec,
                0,
                fitness_scratch + swarm_base + (local_id) * 2
                );
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!local_id && i > 1 && i % 2)// && group_id < num_swarms)
        {
            right = vload2(0, fitness_scratch + swarm_base + (i - 1) * 2);
            right = select(left, max_vec, left < zero);
            updated_vec = fmin(updated_vec, right);
            vstore2(
                updated_vec,
                0,
                fitness_scratch + swarm_base //+ local_id * 2 //this is redundant, since local_id == 0 here
                );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //note: num_sparticles is always divisible by 4, so there is no need to catch extra after the loop

    int update_needed;
    if (!local_id)// && group_id < num_swarms)
    {
        //In the event that no updates were performed, all components of updated_vec will be FLT_MAX - which is guarenteed to be less than or equal to the old sbest fitness (and therefore we don't need to update it)
        update_needed = any(updated_vec != max_vec);
        float final_val = min(updated_vec.x, updated_vec.y);
        
        if ( update_needed && (update_needed = (final_val < sbest_fitness[group_id])) )
        {
            fitness_scratch[swarm_base] = final_val; //min value is now in fitness_scratch[0] for each workgroup
                    
            sbest_fitness[group_id] = final_val;
            group_improved = 1;
        }
        //printf("%u: %f\n", group_id, update_needed ? final_val : -1.0f);
        fitness_scratch[swarm_base + 1] = update_needed; //each workgroup's fitness_scratch[1] now contains a boolean indicating whether or not sbest was updated
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Multiple threads may have fitnesses equal to the minimum fitness. Use a race condition to select one such thread.
    //This thread will write it's corresponding particle's position to sbest_positions.
    update_needed = (int) fitness_scratch[swarm_base + 1];
    if (local_id < num_sparticles * swarms_per_group && update_needed && per_thread_fitness == fitness_scratch[swarm_base])// && group_id < num_swarms)
    {
        fitness_scratch[swarm_base + 2] = (float) local_id; //for each workgroup, fitness_scratch[2] now contains the index of the thread with the updated position
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //fitness_scratch[2] now holds the index of the particle whose fitness is best
    if (update_needed && local_id < num_dims / 4)// && group_id < num_swarms)
    {
        uint available_threads = min(num_dims / 4, get_local_size(0) / swarms_per_group); //in case num_sparticles < num_dims / 4
        uint final_index = (uint) fitness_scratch[swarm_base + 2];
        for (i = 0; i < (num_dims / 4) / available_threads; i++)
        {
            pos_vec = vload4(0, position + group_id * num_sparticles * num_dims + final_index * num_dims + (local_id + i) * 4);
            vstore4(pos_vec, 0, sbest_position + group_id * num_dims + (local_id + i) * 4);
        }

        //clean up any leftover vectors if (available threads) doesn't divide (number of values we need to copy)  evenly
        uint num_leftover = (num_dims / 4) % available_threads;
        if ( local_id < num_leftover )
        {
            pos_vec = vload4(0, position + group_id * num_sparticles * num_dims + final_index * num_dims + ((num_dims / 4 - num_leftover) + local_id) * 4);
            vstore4(pos_vec, 0, sbest_position + group_id * num_dims + ((num_dims / 4 - num_leftover) + local_id) * 4);
        }
    }

    //------------
    //now check the health of the swarms
    
    if (local_id < num_sparticles / 4)// && group_id < num_swarms)
    {
        //if update_occurred is < 0, then no update occurred
        //in updated_vec, a value > 0 means that no update occured
        int4 update_vec = (update_occurred < 0.0f) * (-1);
        update_vec.xy += update_vec.zw;
        partial = (float2) ((float) update_vec.x, (float) update_vec.y);
        vstore2(partial, 0, fitness_scratch + swarm_base + local_id * 2);
    }

    for (i = num_sparticles / 8; i > 0; i /= 2)
    {
        if (local_id < i)// && group_id < num_swarms)
        {
            partial += vload2(0, fitness_scratch + swarm_base + (local_id + i) * 2);
            vstore2(partial, 0, fitness_scratch + swarm_base + local_id * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!local_id && i % 2 && i > 1)// && group_id < num_swarms)
        {
            partial += vload2(0, fitness_scratch + swarm_base + (i - 1) * 2); //+ local_id * 2 is redundant, since local_id == 0 here
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //add in the odd vector, if any
    if ((num_sparticles / 4) > 1 && (num_sparticles / 4) % 2 && !local_id)// && group_id < num_swarms)
    {
        partial += vload2(0, fitness_scratch + swarm_base + ((num_sparticles / 4) - 1) * 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //write one value for each swarm
    if (!local_id)// && group_id < num_swarms)
    {
        float not_updated_ratio = (float) (partial.x + partial.y) / (float) num_sparticles;
        if (!group_improved)
        {
            alg_health[group_id]++;
        }

        //if it's not unhealthy for consecutive iterations, reset to zero
        else
        {
            alg_health[group_id] = 0;
        }
    }
}
