/**
 * This kernel handles updating the position and velocity global memory buffers. It uses the
 * standard PSO equations described in the paper.
 * This kernel is executed using one thread per particle dimension (total threads = num_swarms * num_sparticles * num_tasks / 4).
 */
__kernel void update_pos_vel_hybrid(__global float *positions,               //size is num_swarms * num_sparticles * num_tasks
                             __global float *velocities,              //size is num_swarms * num_sparticles * num_tasks
                             __global float *particle_best_positions, //size is num_swarms * num_sparticles * num_tasks
                             __constant float *swarm_best_positions,    //size is num_swarms * num_tasks
                             __global float *rands,                   //buffer of random numbers of size MAX_NUM_RANDS
                             uint rands_offset,             //index at which to start accessing rands
                             //uint iter_index,
                             __local float *scratch,
                             __constant config *conf                //struct containing simulation parameters
    )
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint num_swarms = conf->num_swarms;
    uint num_sparticles = conf->num_sparticles;
    uint num_tasks = conf->num_dims;
    uint num_machines = conf->num_machines;

    event_t event = async_work_group_copy(scratch,
                          velocities + get_local_size(0) * get_group_id(0),
                          get_local_size(0) * 4,
                          0
        );
    float4 pos_chunk = vload4(0, positions + global_id * 4);
    //float4 vel_chunk = vload4(0, velocities + global_id * 4);

    float4 rands_chunk1 = vload4(0, rands + (rands_offset + global_id * 4));
    float4 rands_chunk2 = vload4(0, rands + (rands_offset + (global_id + 1) * 4));
    float4 pbest_chunk = vload4(0, particle_best_positions + global_id * 4);

    float4 sbest_chunk = vload4(0, swarm_best_positions + global_id / (num_tasks * num_sparticles / 4) + (global_id / 4) * (num_swarms * 4));

    wait_group_events(1, &event);
    //update velocity (using standard PSO equation)
    sbest_chunk = conf->omega * vload4(0, scratch + local_id * 4) +
        conf->c1 * rands_chunk1 *
        (pbest_chunk - pos_chunk) +
        conf->c2 * rands_chunk2 *
        (sbest_chunk - pos_chunk);

    //clamp velocity to [-machines / 2, num_machines / 2]
    sbest_chunk = clamp( sbest_chunk, (float4) (num_machines * -0.5f), (float4) (num_machines * 0.5f) );
    vstore4(sbest_chunk, 0, velocities + global_id * 4);
        
    //update position based on the above velocity.
    //Clamp position to solution space - clamp() function is from OpenCL API
    pos_chunk = clamp( pos_chunk + sbest_chunk, (float4) 0.0f, (float4) (num_machines - 1) );
    vstore4(pos_chunk, 0, positions + global_id * 4);
}
