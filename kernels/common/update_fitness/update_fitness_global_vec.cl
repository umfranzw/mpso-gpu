//num_swarms groups of num_sparticles each
__kernel void update_fitness_global_vec(
    __global float *position,      //size is num_swarms * num_sparticles * num_tasks
    __global float *fitness,       //size is num_swarms * num_sparticles * num_tasks
    __global float *scratch,       //global memory used for scratch - size = num_swarms * num_machines * num_sparticles
    __constant config *conf,     //struct containing simulation parameters
    __constant float *etc_buf,   //ETC matrix (used in makespace computation) size is num_machines * num_tasks
    uint group_size
    )
{
    uint num_sparticles = conf->num_sparticles;
    uint swarms_per_group = group_size / num_sparticles;
    uint local_id = get_local_id(0) % num_sparticles;
    uint group_id = get_group_id(0) * swarms_per_group + (get_local_id(0) / num_sparticles);
    uint num_machines = conf->num_machines;
    uint num_tasks = conf->num_dims;
    //uint num_swarms = conf->num_swarms;

    //if (group_id < num_swarms && local_id < num_sparticles)
    //{
        uint group_pos_base = group_id * (num_sparticles * num_tasks);

        //zero out the scratch array
        uint i;
        for (i = 0; i < num_machines / 4; i++)
        {
            vstore4((float4) 0.0f, 0, scratch + group_id * num_sparticles * num_machines + local_id * num_machines + i * 4);
        }

        float4 makespan = (float4) (-1.0f, -1.0f, -1.0f, -1.0f);
        float4 position_vec;
        uint4 machine_vec;
        uint4 etc_offsets;
        const uint4 dim_offsets = (uint4) (0, 1, 2, 3);
        uint4 scratch_offsets;
        float4 etc_vals;
        float4 scratch_vals;
        float2 temp2;
        float temp;

        for (i = 0; i < num_tasks / 4; i++)
        {
            position_vec = vload4(0, position + group_pos_base + local_id * 4 + i * num_sparticles * 4);
            machine_vec = convert_uint4(position_vec + 0.5f);
            etc_offsets = machine_vec * num_tasks + i * 4 + dim_offsets;
            scratch_offsets = group_id * num_sparticles * num_machines + local_id * num_machines + machine_vec;

            etc_vals = (float4) (etc_buf[etc_offsets.x], etc_buf[etc_offsets.y], etc_buf[etc_offsets.z], etc_buf[etc_offsets.w]);

            //this must be done sequentially to allow combining if two vector elements refer to the same machine
            scratch[scratch_offsets.x] += etc_vals.x;
            scratch[scratch_offsets.y] += etc_vals.y;
            scratch[scratch_offsets.z] += etc_vals.z;
            scratch[scratch_offsets.w] += etc_vals.w;

            scratch_vals = (float4) (scratch[scratch_offsets.x], scratch[scratch_offsets.y], scratch[scratch_offsets.z], scratch[scratch_offsets.w]);
            makespan = fmax(makespan, scratch_vals);
        }

        temp2 = fmax(makespan.xy, makespan.zw);
        temp = fmax(temp2.x, temp2.y);
        barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to sync up to gain benefit of global memory coalescing for final write to fitness array
        fitness[group_id * num_sparticles + local_id] = temp;
        //}
}
