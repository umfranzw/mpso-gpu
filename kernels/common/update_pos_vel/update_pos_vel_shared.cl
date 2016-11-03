/**
 * This kernel handles updating the position and velocity global memory buffers. It uses the
 * standard PSO equations described in the paper.
 * This kernel is executed using one thread per particle dimension (total threads = num_swarms * num_sparticles * num_tasks / 4).
 */
__kernel void update_pos_vel_shared(__global float *positions,               //size is num_swarms * num_sparticles * num_tasks
                             __global float *velocities,              //size is num_swarms * num_sparticles * num_tasks
                             __global float *particle_best_positions, //size is num_swarms * num_sparticles * num_tasks
                             __constant float *swarm_best_positions,    //size is num_swarms * num_tasks
                             __global float *rands,                   //buffer of random numbers of size MAX_NUM_RANDS
                             uint rands_offset,             //index at which to start accessing rands
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

    //update velocity (using standard PSO equation)
    /* new_vel_chunk = conf->omega * vel_chunk + */
    /*     conf->c1 * rands_chunk1 * */
    /*     (pbest_chunk - pos_chunk) + */
    /*     conf->c2 * rands_chunk2 * */
    /*     (sbest_chunk - pos_chunk); */

    event_t event;
    float4 temp1;
    float4 temp2;
    float4 temp3;

    event = async_work_group_copy(scratch,
                                  rands + rands_offset + get_local_size(0) * get_group_id(0) * 4,
                          get_local_size(0) * 4 * 2,
                          0
        );
    
    temp1 = vload4(0, particle_best_positions + global_id * 4);
    uint swarm_offset = global_id / (num_sparticles * num_tasks / 4);
    uint task_offset = global_id / (num_swarms * num_sparticles / 4);
    temp2 = vload4(0, swarm_best_positions + swarm_offset * 4 + task_offset * 4 * num_swarms);
    //temp2 = vload4(0, swarm_best_positions + global_id / (num_tasks * num_sparticles / 4) + (global_id / 4) * (num_swarms * 4));
    temp3 = vload4(0, positions + global_id * 4);
    wait_group_events(1, &event);

    temp1 = conf->c1 * vload4(0, scratch + local_id * 4) * (temp1 - temp3);
    temp2 = conf->c2 * vload4(0, scratch + (local_id + get_local_size(0)) * 4) * (temp2 - temp3);

    temp1 += temp2;
    temp1 += (conf->omega * vload4(0, velocities + global_id * 4));

    temp1 = clamp(temp1,
                  (float4) (num_machines * -0.5f),
                  (float4) (num_machines * 0.5f));

    temp3 = clamp(temp1 + temp3,
                  (float4) 0.0f,
                  (float4) (num_machines - 1));
    
    vstore4(temp1, 0, velocities + global_id * 4);
    vstore4(temp3, 0, positions + global_id * 4);
}
