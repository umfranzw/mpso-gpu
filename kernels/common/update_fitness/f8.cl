__kernel void f8(
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
    
    float4 perm1_chunk;
    float4 perm2_chunk;
    float4 z;
    float2 partial = (float2) (0);
    
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

    if (particle_mask && dim_id < m / 4)
    {
        perm1_chunk = vload4(0, scratch + scratch_particle_offset + dim_id * 4);
        //perm1_chunk = (float4) (1);
    }
    if (particle_mask && dim_id < (num_dims - m) / 4)
    {
        perm2_chunk = vload4(0, scratch + scratch_particle_offset + m + dim_id * 4);
        //perm2_chunk = (float4) (0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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
        /* partial.x += 100.0f * pow( pow(perm1_chunk.z, 2.0f) - perm1_chunk.w, 2.0f ) + pow(perm1_chunk.w - 1, 2.0f); */

        partial.x += 100.0f * pow( pow(perm1_chunk.x, 2.0f) - perm1_chunk.y, 2.0f ) + pow(perm1_chunk.x - 1, 2.0f);
        partial.x += 100.0f * pow( pow(perm1_chunk.y, 2.0f) - perm1_chunk.z, 2.0f ) + pow(perm1_chunk.y - 1, 2.0f);
        partial.x += 100.0f * pow( pow(perm1_chunk.z, 2.0f) - perm1_chunk.w, 2.0f ) + pow(perm1_chunk.z - 1, 2.0f);
    }
    
    //(d - 1) - (m + 1) + 1 = d - m
    if (particle_mask && dim_id < (num_dims - m) / 4)
    {
        //compute sphere function
        perm2_chunk = pow(perm2_chunk, (float4) (2));
        partial.y = perm2_chunk.x + perm2_chunk.y + perm2_chunk.z + perm2_chunk.w;
    }

    if (particle_mask && dim_mask)
    {
        vstore2(partial, 0, scratch + scratch_particle_offset + dim_id * 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //reduction - this keeps the 2 terms separate
    uint i;
    uint num_vals = max(m / 4, (num_dims - m) / 4);
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
    
    //final multiplication on term1, and final add
    if (particle_mask && !dim_id)
    {
        scratch[local_id / threads_per_particle] = partial.x * 1000000.0f + partial.y;
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
