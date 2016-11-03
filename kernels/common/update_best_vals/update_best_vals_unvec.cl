//num_swarms workgroups of 256 (num_sparticles) threads each
__kernel void update_best_vals_unvec(
    __global float *fitness, //size = num_swarms * num_sparticles
    __global float *position, //size = num_swarms * num_sparticles * num_tasks
    __global float *pbest_fitness,   //particle best fitnesses (size = num_swarms * num_sparticles)
    __global float *pbest_position,  //particle best positions (size = num_swarms * num_sparticles * num_tasks)
    __global float *sbest_fitness,   //swarm best fitnesses (size = num_swarms)
    __global float *sbest_position,  //swarm best positions (size = num_swarms * num_tasks)
    __local float *fitness_scratch, //local memory scratch space (size = num_sparticles)
    __constant config *conf        //struct containing simulation parameters
    )
{
    uint num_sparticles = conf->num_sparticles;
    uint num_swarms = conf->num_swarms;
    uint swarms_per_group = get_local_size(0) / num_sparticles;
    uint num_tasks = conf->num_dims;
    uint group_id = get_group_id(0) * swarms_per_group + get_local_id(0) / num_sparticles; //each id corresponds to one swarm (there are num_swarms workgroups in total)
    uint local_id = get_local_id(0) % num_sparticles; //each id corresponds to an offset within a swarm (each workgroup contains num_sparticles threads)
    
    uint position_offset = group_id * num_sparticles * num_tasks + local_id;

    //a subset of threads update the pbest fitnesses
    if (local_id < num_sparticles * swarms_per_group)
    {
        float fitness_el = fitness[group_id * num_sparticles + local_id];
        float pbest_el = pbest_fitness[group_id * num_sparticles + local_id];
        //new_fitness = fitness_el < pbest_el ? fitness_el : pbest_el;
        float new_fitness = -1.0f ;
        if (fitness_el < pbest_el)
        {
            new_fitness = fitness_el;
            pbest_fitness[group_id * num_sparticles + local_id] = new_fitness;
        }

        //update fitness_scratch - store a -1 if no update occurred, and the new fitness value if an update did occur
        fitness_scratch[get_local_id(0)] = new_fitness;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < num_sparticles * swarms_per_group)
    {
        float per_thread_fitness = fitness_scratch[get_local_id(0)];

        //all threads update the pbest positions
        uint i;
        if (per_thread_fitness >= 0)
        {
            for (i = 0; i < num_tasks; i++)
            {
                pbest_position[position_offset + i * num_sparticles] = position[position_offset + i * num_sparticles];
            }
        }

        //perform parallel reduction to find swarm best fitness (use float2s so we can support a swarm size of 4 (using float4s means we need at least 8 particles/swarm for the reduction to work)
        float left;
        float right;
        uint group_swarm_index =  group_id % swarms_per_group;
        uint swarm_base = group_swarm_index * num_sparticles;
        float updated_el;
        for (i = num_sparticles >> 1; i > 0; i >>= 1) //start at num_sparticles / 2
        {
            if (local_id < i)
            {
                left = fitness_scratch[swarm_base + local_id];
                right = fitness_scratch[swarm_base + (local_id + i)];

                left = left < 0.0f ? FLT_MAX : left;
                right = right < 0.0f ? FLT_MAX : right;
                updated_el = fmin(left, right);
                fitness_scratch[swarm_base + local_id] = updated_el;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        int update_needed;
        if (!local_id)
        {
            //In the event that no updates were performed, all components of updated_vec will be FLT_MAX - which is guarenteed to be less than or equal to the old sbest fitness (and therefore we don't need to update it)
            update_needed = (updated_el != FLT_MAX);
            if ( update_needed && (update_needed = (updated_el < sbest_fitness[group_id])) )
            {
                fitness_scratch[swarm_base] = updated_el; //min value is now in fitness_scratch[0] for each workgroup
                    
                sbest_fitness[group_id] = updated_el;
            }
            fitness_scratch[swarm_base + 1] = update_needed; //each workgroup's fitness_scratch[1] now contains a boolean indicating whether or not sbest was updated
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //Multiple threads may have fitnesses equal to the minimum fitness. Use a race condition to select one such thread.
        //This thread will write it's corresponding particle's position to sbest_positions.
        update_needed = (int) fitness_scratch[swarm_base + 1];
        if (update_needed && per_thread_fitness == fitness_scratch[swarm_base])
        {
            fitness_scratch[swarm_base + 2] = (float) local_id; //for each workgroup, fitness_scratch[2] now contains the index of the thread with the updated position
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //fitness_scratch[2] now holds the index of the particle whose fitness is best
        if (update_needed && local_id < num_tasks)
        {
            uint available_threads = min(num_tasks, get_local_size(0) / swarms_per_group); //in case num_sparticles < num_tasks / 4
            uint final_index = (uint) fitness_scratch[swarm_base + 2];
            for (i = 0; i < (num_tasks) / available_threads; i++)
            {
                //sbest_position[group_id * num_tasks + (local_id + i)] = position[group_id * num_sparticles * num_tasks + final_index + (local_id + i) * num_sparticles];
                sbest_position[group_id + (local_id + i) * num_swarms] = position[group_id * num_sparticles * num_tasks + final_index + (local_id + i) * num_sparticles];
            }
        }
    }
}
