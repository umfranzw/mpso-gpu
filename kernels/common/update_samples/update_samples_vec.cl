//num_samples / 4 (or 1) threads
void __kernel update_samples_vec(
    __global float *samples_buf,
    __global float *src_buf, //value to use is assumed to be at index 0
    uint num_samples,
    uint sample_index,
    uint init_output,
    float divisor
    )
{
    uint global_id = get_global_id(0);

    if (!global_id)
    {
        if (init_output)
        {
            samples_buf[sample_index] = src_buf[0];
        }
        else
        {
            samples_buf[sample_index] += src_buf[0];
        }
    }

    if (divisor > 0)
    {
        //it's ok to put this inside an if statement, as long as all threads always take the same branch
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        uint num_vecs = num_samples / 4;
        uint extras = num_samples % 4;

        if (global_id < num_vecs)
        {
            float4 vals = vload4(0, samples_buf + global_id * 4);
            vals /= divisor;
            vstore4(vals, 0, samples_buf + global_id * 4);
        }

        if (global_id < extras) //extras < 4
        {
            samples_buf[num_vecs * 4 + global_id] /= divisor;
        }
    }
}
