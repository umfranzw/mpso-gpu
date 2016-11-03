void __kernel find_min_vec(
    __global float *input_buf,
    __global float *global_scratch, //size = num workgroups, min is left in index 0
    __local float *scratch,
    uint input_len,
    __global float *result_buf,
    uint result_index
    )
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint local_size = get_local_size(0);
    uint num_groups = get_num_groups(0);

    uint num_vecs = input_len / 4;
    uint extras = input_len % 4;
    uint vecs_per_group = num_vecs > local_size ? local_size : num_vecs;
    uint group_vecs = group_id == (num_groups - 1) && (num_vecs % vecs_per_group) ? num_vecs % vecs_per_group : vecs_per_group;
    uint len_mask = local_id < group_vecs;

    float4 vals;
    float2 left;
    float2 right;
    uint i;
    
    if (len_mask)
    {
        vals = vload4(0, input_buf + group_id * vecs_per_group * 4 + local_id * 4);
        left = fmin(vals.hi, vals.lo);

        for (i = local_id; i < extras && group_id == (num_groups - 1); i += local_size)
        {
            left = fmin(left, input_buf[group_id * vecs_per_group * 4 + group_vecs * 4 + i]);
        }
        
        vstore2(left, 0, scratch + local_id * 2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (i = group_vecs / 2; i > 0; i /= 2)
    {
        if (len_mask && local_id < i)
        {
            right = vload2(0, scratch + (local_id + i) * 2);
            left = fmin(left, right);
            vstore2(left, 0, scratch + local_id * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!local_id && i > 1 && i % 2 && len_mask)
        {
            right = vload2(0, scratch + (i - 1) * 2);
            left = fmin(left, right);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //now, thread 0 (for each workgroup) has the minimum
    
    if (local_id == (group_vecs - 1) && group_vecs % 2 && group_vecs > 1)
    {
        vstore2(left, 0, scratch);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (!local_id && group_vecs % 2 && group_vecs > 1)
    {
        left = fmin(left, vload2(0, scratch));
    }

    if (!local_id)
    {
        global_scratch[group_id] = fmin(left.x, left.y);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    //2nd pass done only by group 0
    group_vecs = num_groups / 4;
    extras = num_groups % 4;

    if (!group_id)
    {
        if (local_id < group_vecs)
        {
            vals = vload4(0, global_scratch + local_id * 4);
            left = fmin(vals.lo, vals.hi);
            vstore2(left, 0, scratch + local_id * 2);
        }
        if (local_id < extras)
        {
            scratch[group_vecs * 2 + local_id] = global_scratch[group_vecs * 4 + local_id];
        }
    }
    
    for (i = group_vecs / 2; !group_id && i > 0; i /= 2)
    {
        if (i < local_id)
        {
            right = vload2(0, scratch + (local_id + i) * 2);
            left = fmin(left, right);
            vstore2(left, 0, scratch + local_id * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!local_id && i > 1 && i % 2)
        {
            right = vload2(0, scratch + (i - 1) * 2);
            left = fmin(left, right);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!local_id && !group_id && group_vecs % 2 && group_vecs > 1)
    {
        left = fmin(left, vload2(0, scratch + group_vecs * 2));
    }

    if (!local_id && !group_id && extras)
    {
        for (i = 0; i < extras; i++)
        {
            left.x = fmin(left.x, scratch[group_vecs * 2 + (group_vecs % 2) * 2 + i]);
        }
    }

    if (!local_id && !group_id)
    {
        float result = fmin(left.x, left.y);
        result_buf[result_index] = result;
        //global_scratch[0] = result;
    }
}
