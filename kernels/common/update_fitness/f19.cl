__kernel void f19(
    __global float *positions,  //size = num_swarms * num_sparticles * num_dims
    __global float *fitnesses, //size = num_swarms * num_sparticles
    __global float *optimum, //size = num_dims
    __local float *scratch,
    const uint num_swarms,
    const uint num_sparticles,
    const uint num_dims,
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
    float2 partial = (float2) (0.0f);
    
    if (particle_mask && dim_mask)
    {
        const uint offset = swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4;
        z = vload4(0, positions + offset) - vload4(0, optimum + dim_id * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //compute schwefel function
    if (particle_mask && dim_mask)
    {
        z.yzw += z.xyz;
        z.zw += z.xy;

        //write last value to local mem
        if (!dim_id)
        {
            scratch[scratch_particle_offset] = z.w;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint i;
    for (i = 1; i < num_dims / 4; i++)
    {
        if (dim_id == i)
        {
            z += (float4) scratch[scratch_particle_offset];
            scratch[scratch_particle_offset] = z.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (particle_mask && dim_mask)
    {
        z = pow(z, (float4) (2));
        partial = z.xy + z.zw;
        vstore2(partial, 0, scratch + scratch_particle_offset + dim_id * 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //reduction - this keeps the 2 terms separate
    for (i = num_dims / 8; i > 0; i /= 2)
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
    if ((num_dims / 4) > 1 && (num_dims / 4) % 2 && !dim_id && particle_mask)
    {
        partial += vload2(0, scratch + scratch_particle_offset + ((num_dims / 4) - 1) * 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //write to local, then global, to gain advantage of coalescing
    if (particle_mask && !dim_id)
    {
        scratch[local_id / threads_per_particle] = partial.x + partial.y;
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
