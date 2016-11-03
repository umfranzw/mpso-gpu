//m/4 workgroups, each with m/4 threads
//4 * dim_len local mem elements
//Right now this only works for one work group (max matrix dimensions 1024 x 1024)
//This is not the most efficient method (memory access patterns are at moderate suckage level), but
//this kernel is not used for the actual MPSO alg, so it's good enough for now...
//Also, it will work for any size matrix (as long as dim_len is a multiple of 4, and < 1024), not just powers of 2 like the AMD sample...

void __kernel init_rot_matrix_vec(
    __global float *output,
    __global float *input,
    __local  float *scratch,
    uint dim_len //length of one side
    )
{
    //uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);

    float4 orig_chunk;
    //uint i;
    /* for (i = 0; i < 4; i++) */
    /* { */
    /*     orig_chunk = vload4(0, input + group_id * 4 + (local_id * 4 + i) * dim_len); */
        
    /*     scratch[local_id * 4 + i] = orig_chunk.x; */
    /*     scratch[local_id * 4 + i + dim_len] = orig_chunk.y; */
    /*     scratch[local_id * 4 + i + dim_len * 2] = orig_chunk.z; */
    /*     scratch[local_id * 4 + i + dim_len * 3] = orig_chunk.w; */
    /* } */

    orig_chunk = vload4(0, input + group_id * 4 + (local_id * 4 + 0) * dim_len);    
    scratch[local_id * 4 + 0] = orig_chunk.x;
    scratch[local_id * 4 + 0 + dim_len] = orig_chunk.y;
    scratch[local_id * 4 + 0 + dim_len * 2] = orig_chunk.z;
    scratch[local_id * 4 + 0 + dim_len * 3] = orig_chunk.w;

    orig_chunk = vload4(0, input + group_id * 4 + (local_id * 4 + 1) * dim_len);    
    scratch[local_id * 4 + 1] = orig_chunk.x;
    scratch[local_id * 4 + 1 + dim_len] = orig_chunk.y;
    scratch[local_id * 4 + 1 + dim_len * 2] = orig_chunk.z;
    scratch[local_id * 4 + 1 + dim_len * 3] = orig_chunk.w;

    orig_chunk = vload4(0, input + group_id * 4 + (local_id * 4 + 2) * dim_len);    
    scratch[local_id * 4 + 2] = orig_chunk.x;
    scratch[local_id * 4 + 2 + dim_len] = orig_chunk.y;
    scratch[local_id * 4 + 2 + dim_len * 2] = orig_chunk.z;
    scratch[local_id * 4 + 2 + dim_len * 3] = orig_chunk.w;

    orig_chunk = vload4(0, input + group_id * 4 + (local_id * 4 + 3) * dim_len);    
    scratch[local_id * 4 + 3] = orig_chunk.x;
    scratch[local_id * 4 + 3 + dim_len] = orig_chunk.y;
    scratch[local_id * 4 + 3 + dim_len * 2] = orig_chunk.z;
    scratch[local_id * 4 + 3 + dim_len * 3] = orig_chunk.w;
    
    float4 trans_chunk;
    /* for (i = 0; i < 4; i++) */
    /* { */
    /*     trans_chunk = vload4(0, scratch + i * dim_len + local_id * 4); */
    /*     vstore4(trans_chunk, 0, output + group_id * dim_len * 4 + i * dim_len + local_id * 4); */
    /* } */

    trans_chunk = vload4(0, scratch + 0 * dim_len + local_id * 4);
    vstore4(trans_chunk, 0, output + group_id * dim_len * 4 + 0 * dim_len + local_id * 4);

    trans_chunk = vload4(0, scratch + 1 * dim_len + local_id * 4);
    vstore4(trans_chunk, 0, output + group_id * dim_len * 4 + 1 * dim_len + local_id * 4);

    trans_chunk = vload4(0, scratch + 2 * dim_len + local_id * 4);
    vstore4(trans_chunk, 0, output + group_id * dim_len * 4 + 2 * dim_len + local_id * 4);

    trans_chunk = vload4(0, scratch + 3 * dim_len + local_id * 4);
    vstore4(trans_chunk, 0, output + group_id * dim_len * 4 + 3 * dim_len + local_id * 4);
    
}
