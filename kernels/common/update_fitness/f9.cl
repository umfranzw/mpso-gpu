__kernel void f9(
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
    if (extra_perm2_required && !dim_id) //grab the last 2
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
            //rotate_z function will store to start of scratch + scratch_offset, regardless of z_offset parameter value.
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
            //partial.x = perm1_chunk.x + perm1_chunk.y + perm1_chunk.z + perm1_chunk.w;
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

    //compute elliptic function
    if (particle_mask && dim_id < (num_dims / 2) / 4)
    {
        perm2_chunk = pow(perm2_chunk, ((float4) 2));
        float4 exponent = ( ((float4) dim_id * 4 + (float4) (0, 1, 2, 3)) / (float4) (num_dims / 2 - 1) ) * 6;
        perm2_chunk *= pow(((float4) 10), exponent);

        //store term2 result in partial.y
        //partial.y = perm2_chunk.x + perm2_chunk.y + perm2_chunk.z + perm2_chunk.w;
        partial = perm2_chunk.xy + perm2_chunk.zw;

        //write partial to local mem in preparation for the reduction
        vstore2(partial, 0, scratch + scratch_particle_offset + dim_id * 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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

    if (extra_perm2_required && particle_mask && !dim_id)
    {
        extra_perm2_chunk = pow(extra_perm2_chunk, (float2) (2));
        //this chunk is really "on the end," so 'i' is the highest possible 2 values here
        float2 exponent = ( (float2) (num_dims / 2 - 2, num_dims / 2 - 1) / (float2) (num_dims / 2 - 1) ) * 6;
        extra_perm2_chunk *= pow((float2) 10, exponent);

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
