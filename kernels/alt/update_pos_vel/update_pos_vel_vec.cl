/*
 Global size: s * p * d / 4
 Local size: NULL
 Local memory per thread: 0
 Mapping: One thread per 4 dimensions
*/

__kernel void update_pos_vel_vec(
    __global float *positions,               //s * p * d
    __global float *velocities,              //s * p * d
    __global float *particle_best_positions, //s * p * d
    //__constant float *swarm_best_positions,    //s * d
    __global float *swarm_best_positions,    //s * d
    __constant uint *swarm_types,
    uint num_sparticles,
    uint num_dims,
    float omega,
    float c1,
    float c2,
    float max_axis_val,
    uint seed,
    uint iter_index,
    float max_vel
    )
{
    uint global_id = get_global_id(0);
    uint swarm_id = global_id / (num_sparticles * num_dims / 4);
    uint dim_id = global_id % (num_dims / 4);

    float4 pos_chunk = vload4(0, positions + global_id * 4);
    float4 vel_chunk = vload4(0, velocities + global_id * 4);
    float4 new_vel_chunk;

    float4 r1 = get_float_rands_vec(
        global_id,
        UPDATE_POS_VEL_STREAM,
        iter_index * 2,
        seed
        );
    float4 r2 = get_float_rands_vec(
        global_id,
        UPDATE_POS_VEL_STREAM,
        iter_index * 2 + 1,
        seed
        );

    float4 pbest_chunk = vload4(0, particle_best_positions + global_id * 4);
    float4 sbest_chunk = vload4(0, swarm_best_positions + swarm_id * num_dims + dim_id * 4);

    new_vel_chunk = omega * vel_chunk +
        c1 * r1 *
        (pbest_chunk - pos_chunk) +
        c2 * r2 *
        (sbest_chunk - pos_chunk);

    new_vel_chunk = clamp( new_vel_chunk, (float4) (-1.0f * max_vel), (float4) (max_vel) );
    //store new velocity
    vstore4(new_vel_chunk, 0, velocities + global_id * 4);

    if (swarm_types[swarm_id] == TYPE_PSO)
    {
        //store new position
        float4 new_pos_chunk = pos_chunk + new_vel_chunk;

        //clamp new values to the solution space
        new_pos_chunk = clamp( new_pos_chunk, (float4) (-1.0f * max_axis_val), (float4) (max_axis_val) );
        vstore4(new_pos_chunk, 0, positions + global_id * 4);
    }
}
