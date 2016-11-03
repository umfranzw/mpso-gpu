void permute_z(
    float4 z,
    const uint dim_id,
    const uint particle_mask,
    const uint dim_mask,
    const uint scratch_particle_offset,
    __local float *scratch,
    __global uint *perm_vec
    )
{
    if (particle_mask && dim_mask)
    {
        uint4 perm_chunk = vload4(0, perm_vec + dim_id * 4);

        scratch[scratch_particle_offset + perm_chunk.x] = z.x;
        scratch[scratch_particle_offset + perm_chunk.y] = z.y;
        scratch[scratch_particle_offset + perm_chunk.z] = z.z;
        scratch[scratch_particle_offset + perm_chunk.w] = z.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

void rotate_z(
    uint m,
    uint dim_id,
    uint swarm_id,
    uint particle_mask,
    uint threads_per_particle,
    uint scratch_particle_offset,
    __local float *scratch,
    __global float *rot_matrix,
    uint z_offset
    )
{
    //note: this assumes that rot_matrix has been transposed,
    //So "row" here actually refers to a column, and "col" to a row.
    //We have at least m / 4 threads per particle (at least enough to handle one row of the rotation matrix).
    //We have m^2 / 2 local memory per particle (each thread adds its chunk together to halve the space requirement).

    uint i;
    uint j;
    float4 z_chunk;
    float4 matrix_chunk;
    float2 partial;

    uint row = dim_id / (m / 4);
    uint col = dim_id % (m / 4);
    uint rows_at_once = threads_per_particle / (m / 4);

    if (particle_mask && row < m)
    {
        z_chunk = vload4(0, scratch + scratch_particle_offset + z_offset + col * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE); //at this point, the entire z vector has been read by all threads that need it, so local memory can be overwritten

    //for (i = row; i < m; i += rows_at_once) //i is the current row
    for (i = row; i < row + (m / rows_at_once + (m % rows_at_once ? 1 : 0)) * rows_at_once; i += rows_at_once) //i is the current row
    {
        if (i < m) //this can't go into the loop condition, because all threads need to hit the barrier below
        {
            matrix_chunk = vload4(0, rot_matrix + i * m + col * 4);
            /* if (!swarm_id && !i) */
            /*     printf("[%u: %f, %f]\n", col, matrix_chunk.x, matrix_chunk.y); */

            matrix_chunk *= z_chunk;
            partial = matrix_chunk.lo + matrix_chunk.hi;
            vstore2(partial, 0, scratch + scratch_particle_offset + i * (m / 2) + col * 2);
            /* if (!swarm_id) */
            /*     printf("(%u, %u): %f, %f\n", i, col, partial.x, partial.y); */
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //perform reduction, working on float2 chunks
        for (j = m / 8; j > 0; j /= 2) //j is the index of the current col chunk (float2)
        {
            if (i < m && col < j && particle_mask)
            {
                //partial += vload2(0, scratch + scratch_particle_offset + i * (m / 2) + j * 2);
                partial += vload2(0, scratch + scratch_particle_offset + i * (m / 2) + j * 2 + col * 2);
                vstore2(partial, 0, scratch + scratch_particle_offset + i * (m / 2) + col * 2);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (!col && i < m && j > 1 && j % 2 && particle_mask)
            {
                partial += vload2(0, scratch + scratch_particle_offset + i * (m / 2) + (j - 1) * 2);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        //add in the odd vector, if any
        if (i < m && (m / 4) > 1 && (m / 4) % 2 && !col && particle_mask)
        {
            partial += vload2(0, scratch + scratch_particle_offset + i * (m / 2) + ((m / 4) - 1) * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < m && !col && particle_mask)
        {
            scratch[scratch_particle_offset + i] = partial.x + partial.y;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //now, z is laid out in m consecutive elements at scratch + scratch_particle_offset
}

//----------------

void permute_z2(
    float4 z,
    const uint dim_id,
    const uint scratch_particle_offset,
    __local float *scratch,
    __global uint *perm_vec
    )
{
        uint4 perm_chunk = vload4(0, perm_vec + dim_id * 4);

        scratch[scratch_particle_offset + perm_chunk.x] = z.x;
        scratch[scratch_particle_offset + perm_chunk.y] = z.y;
        scratch[scratch_particle_offset + perm_chunk.z] = z.z;
        scratch[scratch_particle_offset + perm_chunk.w] = z.w;
    barrier(CLK_LOCAL_MEM_FENCE);
}

void rotate_z2(
    uint m,
    uint dim_id,
    uint swarm_id,
    uint threads_per_particle,
    uint scratch_particle_offset,
    __local float *scratch,
    __global float *rot_matrix,
    uint z_offset
    )
{
    //note: this assumes that rot_matrix has been transposed,
    //So "row" here actually refers to a column, and "col" to a row.
    //We have at least m / 4 threads per particle (at least enough to handle one row of the rotation matrix).
    //We have m^2 / 2 local memory per particle (each thread adds its chunk together to halve the space requirement).

    uint i;
    uint j;
    float4 z_chunk;
    float4 matrix_chunk;
    float2 partial;

    uint row = dim_id / (m / 4);
    uint col = dim_id % (m / 4);
    uint rows_at_once = threads_per_particle / (m / 4);

    if (row < m)
    {
        z_chunk = vload4(0, scratch + scratch_particle_offset + z_offset + col * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE); //at this point, the entire z vector has been read by all threads that need it, so local memory can be overwritten

    //for (i = row; i < m; i += rows_at_once) //i is the current row
    for (i = row; i < row + (m / rows_at_once + (m % rows_at_once ? 1 : 0)) * rows_at_once; i += rows_at_once) //i is the current row
    {
        if (i < m) //this can't go into the loop condition, because all threads need to hit the barrier below
        {
            matrix_chunk = vload4(0, rot_matrix + i * m + col * 4);
            /* if (!swarm_id && !i) */
            /*     printf("[%u: %f, %f]\n", col, matrix_chunk.x, matrix_chunk.y); */

            matrix_chunk *= z_chunk;
            partial = matrix_chunk.lo + matrix_chunk.hi;
            vstore2(partial, 0, scratch + scratch_particle_offset + i * (m / 2) + col * 2);
            /* if (!swarm_id) */
            /*     printf("(%u, %u): %f, %f\n", i, col, partial.x, partial.y); */
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //perform reduction, working on float2 chunks
        for (j = m / 8; j > 0; j /= 2) //j is the index of the current col chunk (float2)
        {
            if (i < m && col < j)
            {
                //partial += vload2(0, scratch + scratch_particle_offset + i * (m / 2) + j * 2);
                partial += vload2(0, scratch + scratch_particle_offset + i * (m / 2) + j * 2 + col * 2);
                vstore2(partial, 0, scratch + scratch_particle_offset + i * (m / 2) + col * 2);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (!col && i < m && j > 1 && j % 2)
            {
                partial += vload2(0, scratch + scratch_particle_offset + i * (m / 2) + (j - 1) * 2);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        //add in the odd vector, if any
        if (i < m && (m / 4) > 1 && (m / 4) % 2 && !col)
        {
            partial += vload2(0, scratch + scratch_particle_offset + i * (m / 2) + ((m / 4) - 1) * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < m && !col)
        {
            scratch[scratch_particle_offset + i] = partial.x + partial.y;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //now, z is laid out in m consecutive elements at scratch + scratch_particle_offset
}
