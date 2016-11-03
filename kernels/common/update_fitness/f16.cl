__kernel void f16(
    __global float *positions,  //size = num_swarms * num_sparticles * num_dims
    __global float *fitnesses, //size = num_swarms * num_sparticles
    __global float *optimum, //size = num_dims
    __global float *rot_matrix,
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
    float accum = 0.0f;
    
    if (particle_mask && dim_mask)
    {
        const uint offset = swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4;
        z = vload4(0, positions + offset) - vload4(0, optimum + dim_id * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint k;
    for (k = 0; k < num_dims / m; k++)
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

        rotate_z(
            m,
            dim_id,
            swarm_id,
            particle_mask,
            threads_per_particle,
            scratch_particle_offset,
            scratch,
            rot_matrix,
            k * m
            );

        if (particle_mask && dim_id < m / 4)
        {
            perm1_chunk = vload4(0, scratch + scratch_particle_offset + dim_id * 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //compute rotated ackley function
        if (particle_mask && dim_id < m / 4)
        {
            float4 term1 = pow(perm1_chunk, (float4) 2);
            float4 term2 = native_cos(2 * M_PI_F * perm1_chunk);

            //partial.x is sum of 4 term1 components, partial.y is sum of 4 term 2 components
            partial = (float2) (term1.x + term1.y + term1.z + term1.w,
                                term2.x + term2.y + term2.z + term2.w);
    
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
    
        //final multiplication on term1, and final add
        if (particle_mask && !dim_id)
        {
            //both terms are multiplied by 1 / num_dims
            partial *= (1.0f / (float) m );
            //term1 needs a square root
            partial.x = -0.2f * native_sqrt(partial.x);
            //both sized are exponents of power of e expression
            partial = native_exp(partial);
            //term1 has another multiplier
            partial.x *= -20;

            accum += (partial.x - partial.y + 20 + M_E_F);
        }
    }

    //write to local, then global, to gain advantage of coalescing
    if (particle_mask && !dim_id)
    {
        scratch[local_id / threads_per_particle] = accum;
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
