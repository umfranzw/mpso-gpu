//assume we have input_len / 4 threads, and input_len / 4 local mem
void __kernel find_min(
    __global float *input_buf,
    __global float *output_buf,
    __local float *scratch,
    uint input_len,
    uint output_len,
    uint output_result_index,
    uint init_output,
    float divisor //if <=0, no division is performed
    )
{
    uint global_id = get_global_id(0);

    float4 vals = vload4(0, input_buf + global_id * 4);
    float2 temp = fmin(vals.lo, vals.hi);

    scratch[global_id] = fmin(temp.x, temp.y);

    barrier(CLK_LOCAL_MEM_FENCE);

    uint i;
    for (i = input_len / 16; i > 0; i /= 2)
    {
        if (global_id < i)
        {
            temp = vload2(0, scratch + global_id * 2);
            temp = fmin(temp, vload2(0, scratch + (global_id + i) * 2));
            vstore2(temp, 0, scratch + global_id * 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!global_id)
    {
        uint extra = (input_len / 4) % 4;
        float result = fmin(temp.x, temp.y);

        //this will execute at most 3 times...
        for (i = 0; i < extra; i++)
        {
            result = fmin(result, scratch[input_len / 4 - i - 1]);
        }

        if (!init_output)
        {
            result += output_buf[output_result_index];
        }

        if (divisor > 0)
        {
            result /= divisor;
        }
        
        output_buf[output_result_index] = result;
    }

    
    /* uint global_id = get_global_id(0); */

    /* float4 vals = vload4(0, input_buf + global_id * 4); */
    /* float2 temp = fmin(vals.lo, vals.hi); */

    /* vstore2(temp, 0, scratch + global_id * 2); */
    
    /* barrier(CLK_LOCAL_MEM_FENCE); */

    /* uint i; */
    /* for (i = input_len / 8; i > 0; i /= 2) */
    /* { */
    /*     if (global_id < i) */
    /*     { */
    /*         temp = fmin(temp, vload2(0, scratch + (global_id + i) * 2)); */
    /*         vstore2(temp, 0, scratch + global_id * 2); */
    /*     } */
    /*     barrier(CLK_LOCAL_MEM_FENCE); */
    /* } */

    /* if (get_local_size(0) % 4 && !global_id) */
    /* { */
    /*     temp = fmin(temp, vload2(0, scratch + input_len / 2 - 2)); */
    /* } */

    /* float result; */
    /* if (!global_id) */
    /* { */
    /*     result = fmin(temp.x, temp.y); */
    /*     if (!init_output) */
    /*     { */
    /*         result += output_buf[output_result_index]; */
    /*     } */

    /*     if (divisor > 0) */
    /*     { */
    /*         result /= divisor; */
    /*     } */
    /*     output_buf[output_result_index] = result; */
    /* } */
}
