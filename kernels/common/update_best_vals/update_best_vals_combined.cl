//num_swarms workgroups of 256 (num_sparticles) threads each
__kernel void update_best_vals_combined(
    __global float *fitness, //size = num_swarms * num_sparticles
    __global float *position, //size = num_swarms * num_sparticles * num_tasks
    __global float *pbest_fitness,   //particle best fitnesses (size = num_swarms * num_sparticles)
    __global float *pbest_position,  //particle best positions (size = num_swarms * num_sparticles * num_tasks)
    __global float *sbest_fitness,   //swarm best fitnesses (size = num_swarms)
    __global float *sbest_position,  //swarm best positions (size = num_swarms * num_tasks)
    __local float *fitness_scratch, //local memory scratch space (size = num_sparticles)
    __constant config *conf,        //struct containing simulation parameters

    __global float *velocities,
    __global float *rands,
    uint rands_offset
    )
{
    uint num_machines = conf->num_machines;
    uint num_sparticles = conf->num_sparticles;
    uint num_swarms = conf->num_swarms;
    uint swarms_per_group = get_local_size(0) / num_sparticles;
    uint num_tasks = conf->num_dims;
    uint group_id = get_group_id(0) * swarms_per_group + get_local_id(0) / num_sparticles; //each id corresponds to one swarm (there are num_swarms workgroups in total)
    uint local_id = get_local_id(0) % num_sparticles; //each id corresponds to an offset within a swarm (each workgroup contains num_sparticles threads)
    
    uint position_offset = group_id * num_sparticles * num_tasks + local_id * 4;

    //a subset of threads update the pbest fitnesses
    if (local_id < num_sparticles / 4 && group_id < num_swarms)
    {
        float4 fitness_chunk = vload4(0, fitness + group_id * num_sparticles + local_id * 4);
        float4 pbest_chunk = vload4(0, pbest_fitness + group_id * num_sparticles + local_id * 4);
        int4 cmp = (fitness_chunk < pbest_chunk);
        float4 new_fitnesses = select(pbest_chunk, fitness_chunk, cmp);

        if (any(cmp))
        {
            //store the updated values in pbest_fitnesses
            vstore4( new_fitnesses, 0, pbest_fitness + group_id * num_sparticles + local_id * 4 );
        }

        //update fitness_scratch - store a -1 if no update occurred, and the new fitness value if an update did occur
        float4 temp = select((float4) (-1.0f, -1.0f, -1.0f, -1.0f), new_fitnesses, cmp);
        vstore4( temp, 0, fitness_scratch + (group_id % swarms_per_group) * num_sparticles + local_id * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float per_thread_fitness = fitness_scratch[get_local_id(0)];
    float4 pos_vec;
    uint i;
    if (local_id < num_sparticles * swarms_per_group && group_id < num_swarms)
    {
        //all threads update the pbest positions
        if (per_thread_fitness >= 0)
        {
            for (i = 0; i < num_tasks / 4; i++)
            {
                pos_vec = vload4(0, position + position_offset + i * num_sparticles * 4);
                vstore4(pos_vec, 0, pbest_position + position_offset + i * num_sparticles * 4);
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

    for (i = num_sparticles >> 2; i > 0; i >>= 1) //start at num_sparticles >> 2 because we divide by 2, and then each thread handles 2 elements at a time => 2 * 2 = 4 = 2^2
    {
        if (local_id < i && group_id < num_swarms)
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
    }

    int update_needed;
    if (!local_id && group_id < num_swarms)
    {
        //In the event that no updates were performed, all components of updated_vec will be FLT_MAX - which is guarenteed to be less than or equal to the old sbest fitness (and therefore we don't need to update it)
        update_needed = any(updated_vec != max_vec);
        float final_val = min(updated_vec.x, updated_vec.y);
        
        if ( update_needed && (update_needed = (final_val < sbest_fitness[group_id])) )
        {
            fitness_scratch[swarm_base] = final_val; //min value is now in fitness_scratch[0] for each workgroup
                    
            sbest_fitness[group_id] = final_val;
        }
        fitness_scratch[swarm_base + 1] = update_needed; //each workgroup's fitness_scratch[1] now contains a boolean indicating whether or not sbest was updated
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Multiple threads may have fitnesses equal to the minimum fitness. Use a race condition to select one such thread.
    //This thread will write it's corresponding particle's position to sbest_positions.
    update_needed = (int) fitness_scratch[swarm_base + 1];
    if (local_id < num_sparticles * swarms_per_group && group_id < num_swarms && update_needed && per_thread_fitness == fitness_scratch[swarm_base])
    {
        fitness_scratch[swarm_base + 2] = (float) local_id; //for each workgroup, fitness_scratch[2] now contains the index of the thread with the updated position
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //fitness_scratch[2] now holds the index of the particle whose fitness is best
    if (update_needed && local_id < num_tasks / 4 && group_id < num_swarms)
    {
        uint available_threads = min(num_tasks / 4, get_local_size(0) / swarms_per_group); //in case num_sparticles < num_tasks / 4
        uint final_index = (uint) fitness_scratch[swarm_base + 2];
        for (i = 0; i < (num_tasks / 4) / available_threads; i++)
        {
            pos_vec = vload4(0, position + group_id * num_sparticles * num_tasks + final_index * 4 + (local_id + i) * num_sparticles * 4);
            //vstore4(pos_vec, 0, sbest_position + group_id * num_tasks + (local_id + i) * 4);
            vstore4(pos_vec, 0, sbest_position + group_id * 4 + (local_id + i) * 4 * num_swarms);
        }
    }

    //--------------
    float4 pos_chunk;
    float4 vel_chunk;
    float4 new_vel_chunk;
    float4 rands_chunk1;
    float4 rands_chunk2;
    float4 pbest_chunk;
    uint swarm_offset;
    uint task_offset;
    float4 sbest_chunk;

    for (i = get_global_id(0); i < num_swarms * num_sparticles * num_tasks / 4; i += get_global_size(0))
    {
        pos_chunk = vload4(0, position + i * 4);
        vel_chunk = vload4(0, velocities + i * 4);

        rands_chunk1 = vload4(0, rands + (rands_offset + i * 4));
        rands_chunk2 = vload4(0, rands + (rands_offset + (i + 1) * 4));
        pbest_chunk = vload4(0, pbest_position + i * 4);

        swarm_offset = i / (num_sparticles * num_tasks / 4);
        task_offset = i / (num_swarms * num_sparticles / 4);
        sbest_chunk = vload4(0, sbest_position + swarm_offset * 4 + task_offset * 4 * num_swarms);
        
        //update velocity (using standard PSO equation)
        new_vel_chunk = conf->omega * vel_chunk +
            conf->c1 * rands_chunk1 *
            (pbest_chunk - pos_chunk) +
            conf->c2 * rands_chunk2 *
            (sbest_chunk - pos_chunk);

        //clamp velocity to [-machines / 2, num_machines / 2]
        new_vel_chunk = clamp( new_vel_chunk, (float4) (num_machines * -0.5f), (float4) (num_machines * 0.5f) );
        vstore4(new_vel_chunk, 0, velocities + i * 4);

        //Clamp position to solution space - clamp() function is from OpenCL API
        pos_chunk = clamp( pos_chunk + new_vel_chunk, (float4) 0.0f, (float4) (num_machines - 1) );
        vstore4(pos_chunk, 0, position + i * 4);
    }
}
