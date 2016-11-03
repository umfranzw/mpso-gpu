//Ackley
__kernel void f23(
    __global float *positions,  //size = num_swarms * num_sparticles * num_dims
    __global float *fitnesses, //size = num_swarms * num_sparticles
    __local float *scratch,          //size (per workgroup) = particles_per_group * num_dims / 8
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
        z = vload4(0, positions + global_id * 4);

        float4 term1 = pow(z, ((float4) 2));
        float4 term2 = native_cos(2 * M_PI_F * z);

        //partial.x is sum of 4 term1 components, partial.y is sum of 4 term 2 components
        partial = (float2) (term1.x + term1.y + term1.z + term1.w,
                            term2.x + term2.y + term2.z + term2.w);
    
        vstore2(partial, 0, scratch + local_mem_offset);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //reduction will not mix up the term1 and term2 components of partial
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

    //perform post-sum calculations
    if (mask && !dim_id)
    {
        //both terms are multiplied by 1 / num_dims
        partial *= (1.0f / (float) num_dims);
        //term1 needs a square root
        partial.x = -0.2f * native_sqrt(partial.x);
        //both sized are exponents of power of e expression
        partial = native_exp(partial);
        //term1 has another multiplier
        partial.x *= -20.0f;

        //add 20 + e to the difference of term1 and term2
        scratch[local_id / (num_dims / 4)] = partial.x - partial.y + 20.0f + M_E_F;
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

    /* //write as many float4s as we can */
    /* if (local_id < num_fitnesses / 4) */
    /* { */
    /*     vstore4(vload4(0, scratch + local_id * 4), 0, fitnesses + get_group_id(0) * particles_per_group + local_id * 4); */
    /* } */
    
    /* //must write the remaining fitnesses individually */
    /* if (local_id < num_fitnesses % 4) */
    /* { */
    /*     uint write_offset = (num_fitnesses / 4) * 4; //move past the fitnesses that were already written above */
    /*     fitnesses[get_group_id(0) * particles_per_group + write_offset + local_id] = scratch[write_offset + local_id]; */
    /* } */
}
