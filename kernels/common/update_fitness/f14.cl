__kernel void f14(
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

        //compute rotated elliptic function
        if (particle_mask && dim_id < m / 4)
        {
            perm1_chunk = pow(perm1_chunk, (float4) (2));
            float4 exp_numer = (float4) (dim_id * 4) + (float4) (0, 1, 2, 3);
            float4 exp_denom = (float4) (m - 1);
            float4 exponent = (exp_numer / exp_denom) * 6.0f;
            perm1_chunk *= pow((float4) (10), exponent);
            //store result for term1 in partial.x
            partial = perm1_chunk.xy + perm1_chunk.zw;

            vstore2(partial, 0, scratch + scratch_particle_offset + dim_id * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    
        //reduction
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
            accum += (partial.x + partial.y);
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
