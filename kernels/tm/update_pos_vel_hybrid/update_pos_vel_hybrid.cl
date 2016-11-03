/**
 * This kernel handles updating the position and velocity global memory buffers. It uses the
 * standard PSO equations described in the paper.
 * This kernel is executed using one thread per particle dimension (total threads = num_swarms * num_sparticles * num_tasks / 4).
 */
__kernel void update_pos_vel_hybrid(
    __global float *positions,               //size is num_swarms * num_sparticles * num_tasks
    __global float *velocities,              //size is num_swarms * num_sparticles * num_tasks
    __global float *particle_best_positions, //size is num_swarms * num_sparticles * num_tasks
    __constant float *swarm_best_positions,    //size is num_swarms * num_tasks
    __local float *scratch,
    uint iter_index,
    uint num_swarms,
    uint num_sparticles,
    uint num_dims,
    uint num_machines,
    float omega,
    float c1,
    float c2,
    uint seed
    )
{
    //replace vel with local mem
    /* uint global_id = get_global_id(0); */
    /* uint local_id = get_local_id(0); */
    /* uint num_swarms = num_swarms; */
    /* uint num_sparticles = num_sparticles; */
    /* uint num_tasks = num_dims; */
    /* uint num_machines = num_machines; */

    /* event_t event = async_work_group_copy(scratch, */
    /*                                       velocities + get_group_id(0) * get_local_size(0) * 4, */
    /*                                       get_local_size(0) * 4, */
    /*                                       0 */
    /*     ); */
    
    /* float4 pos_chunk = vload4(0, positions + global_id * 4); */
    /* float4 temp_chunk = vload4(0, particle_best_positions + global_id * 4); */
    /* float4 rands_chunk = vload4(0, rands + (rands_offset + global_id * 4)); */

    /* temp_chunk = c1 * rands_chunk * (temp_chunk - pos_chunk); */
    
    /* rands_chunk = vload4(0, rands + (rands_offset + (global_id + get_global_size(0)) * 4)); */
    /* uint swarm_offset = global_id / (num_sparticles * num_tasks / 4); */
    /* uint task_offset = global_id / (num_swarms * num_sparticles / 4); */
    /* float4 sbest_chunk = vload4(0, swarm_best_positions + swarm_offset * 4 + task_offset * 4 * num_swarms); */
    /* wait_group_events(1, &event); */
    
    /* //temp_chunk += (c2 * rands_chunk * (sbest_chunk - pos_chunk) + omega * vload4(0, scratch + local_id * 4)); */
    /* temp_chunk += (c2 * rands_chunk * (sbest_chunk - pos_chunk) + omega * vload4(0, scratch + local_id * 4)); */
    
    /* //update velocity (using standard PSO equation) */
    /* /\* sbest_chunk = omega * vload4(0, scratch + local_id * 4) + *\/ */
    /* /\*     c1 * rands_chunk1 * *\/ */
    /* /\*     (pbest_chunk - pos_chunk) + *\/ */
    /* /\*     c2 * rands_chunk2 * *\/ */
    /* /\*     (sbest_chunk - pos_chunk); *\/ */

    /* //clamp velocity to [-machines / 2, num_machines / 2] */
    /* temp_chunk = clamp( temp_chunk, (float4) (num_machines * -0.5f), (float4) (num_machines * 0.5f) ); */
    /* vstore4(temp_chunk, 0, velocities + global_id * 4); */
        
    /* //update position based on the above velocity. */
    /* //Clamp position to solution space - clamp() function is from OpenCL API */
    /* pos_chunk = clamp( pos_chunk + temp_chunk, (float4) 0.0f, (float4) (num_machines - 1) ); */
    /* vstore4(pos_chunk, 0, positions + global_id * 4); */
    

    //-------------------------
    //replace pos with local mem
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    event_t event = async_work_group_copy(scratch,
                          positions + get_local_size(0) * get_group_id(0) * 4,
                          get_local_size(0) * 4,
                          0
        );
    //float4 pos_chunk = vload4(0, positions + global_id * 4);
    float4 vel_chunk = vload4(0, velocities + global_id * 4);

    float4 rands_chunk1 = get_float_rands_vec(
        global_id,
        UPDATE_POS_VEL_STREAM,
        iter_index * 2,
        seed
        );
    float4 rands_chunk2 = get_float_rands_vec(
        global_id,
        UPDATE_POS_VEL_STREAM,
        iter_index * 2 + 1,
        seed
        );

    float4 pbest_chunk = vload4(0, particle_best_positions + global_id * 4);

    uint swarm_offset = global_id / (num_sparticles * num_dims / 4);
    uint task_offset = global_id / (num_swarms * num_sparticles / 4);
    float4 sbest_chunk = vload4(0, swarm_best_positions + swarm_offset * 4 + task_offset * 4 * num_swarms);
    
    wait_group_events(1, &event);
    //update velocity (using standard PSO equation)
    vel_chunk = omega * vel_chunk +
        c1 * rands_chunk1 *
        (pbest_chunk - vload4(0, scratch + local_id * 4)) +
        c2 * rands_chunk2 *
        (sbest_chunk - vload4(0, scratch + local_id * 4));

    //clamp velocity to [-machines / 2, num_machines / 2]
    vel_chunk = clamp( vel_chunk, (float4) (num_machines * -0.5f), (float4) (num_machines * 0.5f) );
    vstore4(vel_chunk, 0, velocities + global_id * 4);
        
    //update position based on the above velocity.
    //Clamp position to solution space - clamp() function is from OpenCL API
    vel_chunk = clamp( vload4(0, scratch + local_id * 4) + vel_chunk, (float4) 0.0f, (float4) (num_machines - 1) );
    vstore4(vel_chunk, 0, positions + global_id * 4);
}
