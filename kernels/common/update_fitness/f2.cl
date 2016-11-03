__kernel void f2(
    __global float *positions,  //size = num_swarms * num_sparticles * num_dims
    __global float *fitnesses, //size = num_swarms * num_sparticles
    __local float *scratch,          //size (per workgroup) = particles_per_group * num_dims / 8
    __global float *optimum, //size = num_dims
    const uint num_swarms,
    const uint num_sparticles,
    const uint num_dims
    )
{
    const uint global_id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint particles_per_group = get_local_size(0) / (num_dims / 4);
    const uint swarm_id = global_id / (num_sparticles * num_dims / 4);
    const uint particle_id = (global_id / (num_dims / 4) ) % num_sparticles;
    const uint dim_id = global_id % (num_dims / 4);
    const uint local_mem_offset = ((swarm_id * num_sparticles + particle_id) % particles_per_group) * (num_dims / 4) * 2 + dim_id * 2;
    const uint mask = swarm_id * num_sparticles + particle_id < num_swarms * num_sparticles;

    float4 z;
    float2 partial;

    //math function code and 2-way component-wise reduction, write results to local mem
    if (mask)
    {
        z = vload4(0, positions + global_id * 4) - vload4(0, optimum + dim_id * 4);
        
        z = pow(z, (float4) 2) - 10 * native_cos(2 * M_PI_F * z) + 10;
        partial = z.xy + z.zw;
        
        vstore2(partial, 0, scratch + local_mem_offset);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //perform reduction, working on float2 chunks
    uint i;
    for (i = num_dims / 8; i > 0; i /= 2)
    {
        if (dim_id < i && mask)
        {
            partial += vload2(0, scratch + local_mem_offset + i * 2);
            vstore2(partial, 0, scratch + local_mem_offset);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!dim_id && i > 1 && i % 2 && mask)
        {
            partial += vload2(0, scratch + local_mem_offset + (i - 1) * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //add in the odd vector, if any
    if ((num_dims / 4) > 1 && (num_dims / 4) % 2 && !dim_id && mask)
    {
        partial += vload2(0, scratch + local_mem_offset + ((num_dims / 4) - 1) * 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //perform final component-wise reduction
    if (mask && !dim_id)
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
