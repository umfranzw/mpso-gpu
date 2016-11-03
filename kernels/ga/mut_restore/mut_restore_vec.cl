//s * p * d / 4 threads
void kernel mut_restore_vec(
    __global float *pre_mut_fit,
    __global float *pre_mut_pos,
    __global float *pre_mut_vel,
    __global float *fitness,
    __global float *positions,
    __global float *velocities,
    __global uint *swarm_health,
    __local int *scratch, //particles_per_group
    uint num_swarms,
    uint num_sparticles,
    uint num_dims,
    uint unhealthy_iters
    )
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint particles_per_group = get_local_size(0) / (num_dims / 4);

    if (local_id < particles_per_group / 2) //we have at least particles_per_group threads, so this will cover the whole local memory buffer
    {
        vstore2((int2) (0), 0, scratch + local_id * 2);
    }

    //note: while we could access more than particles_per_group fitnesses, we only have local memory for particles_per_group fitnesses (can't access another workgroup's local memory)
    //Here, mapping is one thread per 4 particles

    uint num_vecs = particles_per_group / 4;
    uint leftover = particles_per_group % 4;
    if (local_id < num_vecs)
    {
        uint abs_particle_offset = group_id * particles_per_group + local_id * 4;
        uint4 swarm_ids = ( (uint4) abs_particle_offset + (uint4) (0, 1, 2, 3) ) / (uint4) num_sparticles;
        uint4 health_vec = (uint4) (
            swarm_health[swarm_ids.x],
            swarm_health[swarm_ids.y],
            swarm_health[swarm_ids.z],
            swarm_health[swarm_ids.w]
            );
        
        int4 health_cmp = health_vec >= (uint4) (unhealthy_iters);
        if (any(health_cmp))
        {
            float4 prev_fitness = vload4(0, pre_mut_fit + abs_particle_offset);
            float4 cur_fitness = vload4(0, fitness + abs_particle_offset);

            int4 fitness_cmp = (cur_fitness > prev_fitness); // -1 if cur_fitness is worse
            int4 cmp = health_cmp && fitness_cmp;
            if (any(cmp))
            {
                //restore old fitnesses
                float4 updated_fitness = select(cur_fitness, prev_fitness, cmp);
                vstore4(updated_fitness, 0, fitness + abs_particle_offset);

                //convert cmp into two bit vectors instead of this?
                vstore2(-1 * cmp.xy, 0, scratch + local_id * 4);
                vstore2(-1 * cmp.zw, 0, scratch + local_id * 4 + 2);
            }
        }
    }
    
    if (local_id < leftover)
    {
        uint abs_particle_offset = group_id * particles_per_group + num_vecs * 4 + local_id;
        uint swarm_id = abs_particle_offset / num_sparticles;
        uint health = swarm_health[swarm_id];
        int health_cmp = health >= unhealthy_iters;

        if (health_cmp)
        {
            float prev_fitness = pre_mut_fit[abs_particle_offset];
            float cur_fitness = fitness[abs_particle_offset];

            int cmp = health_cmp && (cur_fitness > prev_fitness); //1 if cur_fitness is worse

            if (cmp)
            {
                fitness[abs_particle_offset] = prev_fitness;
                scratch[num_vecs * 4 + local_id] = cmp;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Here, mapping is one thread per 4 dimensions (i.e. num_dims / 4 threads per particle)
    uint rel_particle_id = local_id / (num_dims / 4);
    uint abs_particle_id = group_id * particles_per_group + rel_particle_id;
    uint swarm_id = abs_particle_id / num_sparticles;

    uint particle_id = abs_particle_id % num_sparticles;
    uint dim_id = local_id % (num_dims / 4); //note: the threads for one particle will never be split across two workgroups

    if (swarm_health[swarm_id] >= unhealthy_iters)
    {
        //we could move this outside the loop, since there is a valid local mem location for every particle - however, this would result in tons of bank conflicts.
        //Filtering things down to only unhealthy swarms first (via the above if statement) will seriously reduce the number of local memory accesses we need (for the cost of one global memory access).
        //if (scratch[particle_id])
        if (scratch[abs_particle_id % particles_per_group])
        {
            uint offset = swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4;
            vstore4( vload4(0, pre_mut_pos + offset), 0, positions + offset );
            vstore4( vload4(0, pre_mut_vel + offset), 0, velocities + offset );
        }
    }
    
    if (global_id < num_swarms / 4)
    {
        uint4 health_vec = vload4(0, swarm_health + global_id * 4);
        health_vec = select(health_vec, (uint4) (0), health_vec >= unhealthy_iters);
        vstore4(health_vec, 0, swarm_health + global_id * 4);
    }
}
