//Sphere
//Threads:       d/4
//Local mem:  (d / 4) * 2
__kernel void f21(
    __global float *positions,  //size = num_swarms * num_sparticles * num_dims
    __global float *fitnesses, //size = num_swarms * num_sparticles
    __local float *scratch,          //size (per workgroup) = particles_per_group * num_dims / 8
    uint num_swarms,
    uint num_sparticles,
    uint num_dims
    )
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint particles_per_group = get_local_size(0) / (num_dims / 4); //guarenteed to be >= 1
    uint swarm_id = global_id / (num_sparticles * num_dims / 4);
    uint particle_id = (global_id / (num_dims / 4) ) % num_sparticles;
    uint dim_id = global_id % (num_dims / 4);
    uint local_mem_offset = ((swarm_id * num_sparticles + particle_id) % particles_per_group) * (num_dims / 4) * 2 + dim_id * 2;
    uint mask = swarm_id * num_sparticles + particle_id < num_swarms * num_sparticles;

    float4 z;
    float2 partial;

    //math function code and 2-way component-wise reduction, write results to local mem
    if (mask)
    {
        z = vload4(0, positions + global_id * 4);
        
        z = pow(z, ((float4) 2));

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
    
    //perform final component-wise reduction, store to local mem to take advantage of coalescing below
    if (mask && !dim_id)
    {
        scratch[local_id / (num_dims / 4)] = partial.x + partial.y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint num_fitnesses = ((get_group_id(0) == get_num_groups(0) - 1) && ((num_swarms * num_sparticles) % particles_per_group)) ?
        (num_swarms * num_sparticles) % particles_per_group : particles_per_group;

    async_work_group_copy(
        fitnesses + get_group_id(0) * particles_per_group,
        scratch,
        num_fitnesses,
        0
        );
}
