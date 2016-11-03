/*
 Global size: s * p
 Local size: p
 Local memory per thread: num_machines
 Mapping: One thread per particle, loop through 4 dimensions at a time
*/

__kernel void update_fitness_shared_vec(
    __global float *position,      //s * p * d
    __global float *fitness,       //s * p
    __local float *scratch,       //num_machines * p
    __constant float *etc_buf,   //num_machines * d
    uint swarms_per_group,
    uint num_sparticles,
    uint num_swarms,
    uint num_machines,
    uint num_dims
    )
{
    uint local_id = get_local_id(0) % num_sparticles;
    uint group_id = get_group_id(0) * swarms_per_group + (get_local_id(0) / num_sparticles);

    uint group_pos_base = group_id * (num_sparticles * num_dims);
    uint local_mem_offset = (get_local_id(0) / num_sparticles) * num_sparticles * num_machines;

    uint i;
    //zero out the scratch array
    //Approach #1
    for (i = get_local_id(0) * 2; i < num_machines * swarms_per_group * num_sparticles; i += (get_local_size(0) * 2))
    {
        vstore2((float2) 0.0f, 0, scratch + i);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Approach #2
    //move the zeroing into loop below using an if statement (if first iteration)
    //This reduces register count by 1, which is not enough to raise occupancy. However, it does shift the limiting factor from VGPRs to local memory.
    //Reducing the workgroup size (number of swarms per group) does not help.
    //This approach is slightly slower because of the conditional statements that must be run every time the loop iterates.

    float4 makespan = (float4) (-1.0f, -1.0f, -1.0f, -1.0f);
    float4 position_vec;
    uint4 machine_vec;
    uint4 etc_offsets;
    const uint4 dim_offsets = (uint4) (0, 1, 2, 3);
    uint4 scratch_offsets;
    float4 etc_vals;
    float4 scratch_vals;

    for (i = 0; i < num_dims / 4 && group_id < num_swarms; i++)
    {
        position_vec = vload4(0, position + group_pos_base + local_id * 4 + i * num_sparticles * 4);
        machine_vec = convert_uint4(position_vec + 0.5f);
        scratch_offsets = ((uint4) local_mem_offset + local_id * num_machines) + machine_vec;
        etc_offsets = machine_vec * num_dims + i * 4 + dim_offsets;

        etc_vals = (float4) (etc_buf[etc_offsets.x], etc_buf[etc_offsets.y], etc_buf[etc_offsets.z], etc_buf[etc_offsets.w]);

        //this must be done sequentially to allow combining if two vector elements refer to the same machine
        //stage 4 - this reduces bank conflicts considerably over stage3, but requires considerably more ALU ops. The benefit (reduced memory accesses) from searching down the ternary expressions is not worth the penalty incurred by the extra ALU instructions that are required to do so. (in other words, stage 3 still takes less time, on average, than stage 4)
        /* scratch_vals.x = scratch[scratch_offsets.x] + etc_vals.x; */
        /* scratch[scratch_offsets.x] = scratch_vals.x; */
        
        /* scratch_vals.y = (scratch_offsets.y == scratch_offsets.x ? scratch_vals.x : scratch[scratch_offsets.y]) + etc_vals.y; */
        /* scratch[scratch_offsets.y] = scratch_vals.y; */
        
        /* scratch_vals.z = (scratch_offsets.z == scratch_offsets.y ? */
        /*                   scratch_vals.y : */
        /*                   scratch_offsets.z == scratch_offsets.x ? */
        /*                    scratch_vals.x : */
        /*                    scratch[scratch_offsets.z]) + etc_vals.z; */
        /* scratch[scratch_offsets.z] = scratch_vals.z; */

        /* scratch_vals.w = (scratch_offsets.w == scratch_offsets.z ? */
        /*                   scratch_vals.z : */
        /*                   scratch_offsets.w == scratch_offsets.y ? */
        /*                    scratch_vals.y : */
        /*                    scratch_offsets.w == scratch_offsets.x ? */
        /*                     scratch_vals.x : */
        /*                     scratch[scratch_offsets.w]) + etc_vals.w; */
        /* scratch[scratch_offsets.w] = scratch_vals.w; */
        

        //stage 3
        /* scratch_vals.x = scratch[scratch_offsets.x] + etc_vals.x; */
        /* scratch[scratch_offsets.x] = scratch_vals.x; */
        
        /* scratch_vals.y = (scratch_offsets.x == scratch_offsets.y ? scratch_vals.x + etc_vals.y : scratch[scratch_offsets.y] + etc_vals.y); */
        /* scratch[scratch_offsets.y] = scratch_vals.y; */
        
        /* scratch_vals.z = (scratch_offsets.y == scratch_offsets.z ? scratch_vals.y + etc_vals.z: scratch[scratch_offsets.z] + etc_vals.z); */
        /* scratch[scratch_offsets.z] = scratch_vals.z; */
        
        /* scratch_vals.w = (scratch_offsets.z == scratch_offsets.w ? scratch_vals.z + etc_vals.w : scratch[scratch_offsets.w] + etc_vals.w); */
        /* scratch[scratch_offsets.w] = scratch_vals.w; */

        //stage 2
        scratch_vals.x = scratch[scratch_offsets.x] + etc_vals.x;
        scratch[scratch_offsets.x] = scratch_vals.x;
        scratch_vals.y = scratch[scratch_offsets.y] + etc_vals.y;
        scratch[scratch_offsets.y] = scratch_vals.y;
        scratch_vals.z = scratch[scratch_offsets.z] + etc_vals.z;
        scratch[scratch_offsets.z] = scratch_vals.z;
        scratch_vals.w = scratch[scratch_offsets.w] + etc_vals.w;
        scratch[scratch_offsets.w] = scratch_vals.w;

        //stage 1
        /* scratch[scratch_offsets.x] += etc_vals.x; */
        /* scratch[scratch_offsets.y] += etc_vals.y; */
        /* scratch[scratch_offsets.z] += etc_vals.z; */
        /* scratch[scratch_offsets.w] += etc_vals.w; */

        /* scratch_vals.x = scratch[scratch_offsets.x]; */
        /* scratch_vals.y = scratch[scratch_offsets.y]; */
        /* scratch_vals.z = scratch[scratch_offsets.z]; */
        /* scratch_vals.w = scratch[scratch_offsets.w]; */

        /* scratch_vals = (float4) (scratch[scratch_offsets.x], scratch[scratch_offsets.y], scratch[scratch_offsets.z], scratch[scratch_offsets.w]); */
        makespan = fmax(makespan, scratch_vals);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (group_id < num_swarms)
    {
        makespan.xy = fmax(makespan.xy, makespan.zw);
        makespan.x = fmax(makespan.x, makespan.y);
        
        fitness[group_id * num_sparticles + local_id] = makespan.x;
    }
}
