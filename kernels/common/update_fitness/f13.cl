__kernel void f13(
    __global float *positions,  //size = num_swarms * num_sparticles * num_dims
    __global float *fitnesses, //size = num_swarms * num_sparticles
    __global float *optimum, //size = num_dims
    __global uint *perm_vec,
    __local float *scratch,
    const uint num_swarms,
    const uint num_sparticles,
    const uint num_dims,
    const uint m,
    const uint threads_per_particle,
    const uint scratch_per_particle
    )
{
    const uint global_id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint particles_per_group = get_local_size(0) / threads_per_particle;
    const uint swarm_id = global_id / (num_sparticles * threads_per_particle);
    const uint particle_id = (global_id / threads_per_particle) % num_sparticles;
    const uint dim_id = global_id % threads_per_particle;
    const uint particle_mask = swarm_id * num_sparticles + particle_id < num_swarms * num_sparticles;
    const uint dim_mask = dim_id < (num_dims / 4);
    const uint scratch_particle_offset = ( (swarm_id * num_sparticles + particle_id) % (particles_per_group) ) * scratch_per_particle;
    
    float4 z;
    float2 partial;
    float4 perm1_chunk;
    float4 perm2_chunk;
    float2 extra_perm2_chunk;
    uint extra_perm2_required;
    float accum = 0.0f;
    
    if (particle_mask && dim_mask)
    {
        const uint offset = swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4;
        z = vload4(0, positions + offset) - vload4(0, optimum + dim_id * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //compute z(P_1 : P_m)
    permute_z(
        z,
        dim_id,
        particle_mask,
        dim_mask,
        scratch_particle_offset,
        scratch,
        perm_vec
        );

    if (particle_mask && dim_id < (num_dims / 2) / 4) //this will round down. The remainder are picked up by the case below.
    {
        perm2_chunk = vload4(0, scratch + scratch_particle_offset + (num_dims / 2) + dim_id * 4);
    }
    extra_perm2_required = (num_dims / 2) % 4; //the only time that this can happen, (num_dims / 2) % 4 = 2
    if (extra_perm2_required  && !dim_id) //grab the last 2
    {
        extra_perm2_chunk = vload2(0, scratch + scratch_particle_offset + num_dims - 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint k;
    for (k = 0; k < num_dims / (2 * m); k++)
    {
        partial = (float2) (0.0f);

        //compute z(P_1 : P_m)
        permute_z(
            z,
            dim_id,
            particle_mask,
            dim_mask,
            scratch_particle_offset,
            scratch,
            perm_vec
            );
        
        if (particle_mask && dim_id < m / 4)
        {
            perm1_chunk = vload4(0, scratch + scratch_particle_offset + (k * m) + dim_id * 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //compute rosenbrock function
        if (particle_mask && dim_id < m / 4 - 1) //last thread need not write
        {
            //write last value for next thread to pick up
            scratch[scratch_particle_offset + dim_id] = perm1_chunk.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (particle_mask && dim_id > 0 && dim_id < m / 4)
        {
            float last_z = scratch[scratch_particle_offset + dim_id - 1];

            //partial.x = 100.0f * pow( pow(last_z, 2.0f) - perm1_chunk.x, 2.0f ) + pow(perm1_chunk.x - 1, 2.0f);

            partial.x = 100.0f * pow( pow(last_z, 2.0f) - perm1_chunk.x, 2.0f ) + pow(last_z - 1, 2.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (particle_mask && dim_id < m / 4)
        {
             /* partial.x += 100.0f * pow( pow(perm1_chunk.x, 2.0f) - perm1_chunk.y, 2.0f ) + pow(perm1_chunk.y - 1, 2.0f); */
            /* partial.x += 100.0f * pow( pow(perm1_chunk.y, 2.0f) - perm1_chunk.z, 2.0f ) + pow(perm1_chunk.z - 1, 2.0f); */
            /* partial.y += 100.0f * pow( pow(perm1_chunk.z, 2.0f) - perm1_chunk.w, 2.0f ) + pow(perm1_chunk.w - 1, 2.0f); */

            partial.x += 100.0f * pow( pow(perm1_chunk.x, 2.0f) - perm1_chunk.y, 2.0f ) + pow(perm1_chunk.x - 1, 2.0f);
            partial.x += 100.0f * pow( pow(perm1_chunk.y, 2.0f) - perm1_chunk.z, 2.0f ) + pow(perm1_chunk.y - 1, 2.0f);
            partial.y += 100.0f * pow( pow(perm1_chunk.z, 2.0f) - perm1_chunk.w, 2.0f ) + pow(perm1_chunk.z - 1, 2.0f);
            
            vstore2(partial, 0, scratch + scratch_particle_offset + dim_id * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        //reduction - this keeps the 2 terms separate
        uint num_vals = m / 4;
        uint i;
        for (i = num_vals / 2; i > 0; i /= 2)
        {
            if (dim_id < i && particle_mask && dim_mask)
            {
                partial += vload2(0, scratch + scratch_particle_offset + dim_id * 2 + i * 2);
                vstore2(partial, 0, scratch + scratch_particle_offset + dim_id * 2);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (!dim_id && i > 1 && i % 2 && particle_mask && dim_mask)
            {
                partial += vload2(0, scratch + scratch_particle_offset + (i - 1) * 2);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        //add in the odd vector, if any
        if (num_vals > 1 && num_vals % 2 && !dim_id && particle_mask && dim_mask)
        {
            partial += vload2(0, scratch + scratch_particle_offset + (num_vals - 1) * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //final add
        if (particle_mask && !dim_id)
        {
            accum += (partial.x + partial.y);
        }
    }

    if (particle_mask && dim_id < (num_dims / 2) / 4)
    {
        //compute sphere function
        perm2_chunk = pow(perm2_chunk, (float4) (2));
        //partial.y = perm2_chunk.x + perm2_chunk.y + perm2_chunk.z + perm2_chunk.w;
        partial = perm2_chunk.xy + perm2_chunk.zw;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //reduction - this keeps the 2 terms separate
    uint num_vals = (num_dims / 2) / 4;
    uint i;
    for (i = num_vals / 2; i > 0; i /= 2)
    {
        if (dim_id < i && particle_mask && dim_mask)
        {
            partial += vload2(0, scratch + scratch_particle_offset + dim_id * 2 + i * 2);
            vstore2(partial, 0, scratch + scratch_particle_offset + dim_id * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!dim_id && i > 1 && i % 2 && particle_mask && dim_mask)
        {
            partial += vload2(0, scratch + scratch_particle_offset + (i - 1) * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //add in the odd vector, if any
    if (num_vals > 1 && num_vals % 2 && !dim_id && particle_mask && dim_mask)
    {
        partial += vload2(0, scratch + scratch_particle_offset + (num_vals - 1) * 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (extra_perm2_required && !dim_id)
    {
        extra_perm2_chunk = pow(extra_perm2_chunk, (float2) (2));
            
        partial += extra_perm2_chunk;
    }

    //write to local, then global, to gain advantage of coalescing
    if (particle_mask && !dim_id)
    {
        scratch[local_id / threads_per_particle] = accum + partial.x + partial.y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //store to global memory fitnesses buffer
    uint num_fitnesses = particles_per_group;
    if ((get_group_id(0) == get_num_groups(0) - 1) && ((num_swarms * num_sparticles) % particles_per_group))
    {
        num_fitnesses = (num_swarms * num_sparticles) % particles_per_group;
    }

    async_work_group_copy(
        fitnesses + get_group_id(0) * particles_per_group,
        scratch,
        num_fitnesses,
        0
        );
}
