//Rosenbrock
__kernel void f24(
    __global float *positions,  //size = num_swarms * num_sparticles * num_dims
    __global float *fitnesses, //size = num_swarms * num_sparticles
    __local float *scratch,
    const uint num_swarms,
    const uint num_sparticles,
    const uint num_dims
    )
{
    const uint global_id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint particles_per_group = get_local_size(0) / (num_dims / 4);
    const uint swarm_id = global_id / (num_sparticles * num_dims / 4);
    const uint particle_id = (global_id / (num_dims / 4)) % num_sparticles;
    const uint dim_id = global_id % (num_dims / 4);
    const uint particle_mask = swarm_id * num_sparticles + particle_id < num_swarms * num_sparticles;
    const uint dim_mask = dim_id < (num_dims / 4);
    const uint scratch_particle_offset = ( (swarm_id * num_sparticles + particle_id) % (particles_per_group) ) * (num_dims / 4) * 2;
    
    float4 z;
    float2 partial = (float2) (0);
    
    if (particle_mask && dim_mask)
    {
        const uint offset = swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4;
        z = vload4(0, positions + offset);
        //z = (float4) (1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (particle_mask && dim_id < num_dims / 4 - 1) //last thread need not write
    {
        //write last value for next thread to pick up
        scratch[scratch_particle_offset + dim_id] = z.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (particle_mask && dim_id > 0 && dim_id < num_dims / 4)
    {
        float last_z = scratch[scratch_particle_offset + dim_id - 1];

        //partial.x = 100.0f * pow( pow(last_z, 2.0f) - z.x, 2.0f ) + pow(z.x - 1, 2.0f);
        partial.x = 100.0f * pow( pow(last_z, 2.0f) - z.x, 2.0f ) + pow(last_z - 1, 2.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (particle_mask && dim_id < num_dims / 4)
    {
        /* partial.y += 100.0f * pow( pow(z.x, 2.0f) - z.y, 2.0f ) + pow(z.y - 1, 2.0f); */
        /* partial.y += 100.0f * pow( pow(z.y, 2.0f) - z.z, 2.0f ) + pow(z.z - 1, 2.0f); */
        /* partial.y += 100.0f * pow( pow(z.z, 2.0f) - z.w, 2.0f ) + pow(z.w - 1, 2.0f); */

        partial.y += 100.0f * pow( pow(z.x, 2.0f) - z.y, 2.0f ) + pow(z.x - 1, 2.0f);
        partial.y += 100.0f * pow( pow(z.y, 2.0f) - z.z, 2.0f ) + pow(z.y - 1, 2.0f);
        partial.y += 100.0f * pow( pow(z.z, 2.0f) - z.w, 2.0f ) + pow(z.z - 1, 2.0f);
    }
    
    if (particle_mask && dim_mask)
    {
        vstore2(partial, 0, scratch + scratch_particle_offset + dim_id * 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //perform reduction, working on float2 chunks
    uint i;
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
    
    //perform final component-wise reduction, store to local mem to take advantage of coalescing below
    if (particle_mask && !dim_id)
    {
        scratch[local_id / (num_dims / 4)] = partial.x + partial.y;
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
