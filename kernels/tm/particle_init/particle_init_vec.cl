/*
 Global size: s * p * d / 4
 Local size: NULL
 Local memory per thread: 0
 Mapping: One thread per 4 dimensions
*/

__kernel void particle_init_vec(
    __global float *positions,       //s * p * d
    __global float *velocities,      //s * p * d
    __global float *pbest_positions, //s * p * d
    __global float *pbest_fitnesses, //s * p
    __global float *sbest_positions, //s * d
    __global float *sbest_fitnesses, //s
    const uint num_swarms,
    const uint num_sparticles,
    const uint num_dims,
    const uint max_axis_val,
    const uint seed
    )
{
    const uint global_id = get_global_id(0);

    //random numbers come in the range [0, 1] - we need to expand the range as follows:
    //positions: [0, max_axis_val]
    //velocities: [-max_axis_val / 4, max_axis_val / 4]
    //Each thread will use 8 random numbers - one float4 for setting its particle's position dimension and
    //one float4 for setting its particle's velocity dimension.
        
    float4 rands = get_float_rands_vec(
        global_id,
        PARTICLE_INIT_STREAM,
        0, //this is only called once per iteration, so a constant is fine here. The seed will ensure different values across reps.
        seed
        );
    
    vstore4(rands * max_axis_val, 0, positions + global_id * 4);

    rands = get_float_rands_vec(
        global_id,
        PARTICLE_INIT_STREAM,
        1,
        seed
        );

    vstore4(rands * (max_axis_val / 4.0f) * 2.0f - (max_axis_val / 4.0f), 0, velocities + global_id * 4);

    //initialize everything else to the max float value
    vstore4((float4) (FLT_MAX), 0, pbest_positions + global_id * 4);

    if (global_id < num_swarms * num_sparticles / 4)
    {
        vstore4((float4) (FLT_MAX), 0, pbest_fitnesses + global_id * 4);
    }

    if (global_id < num_swarms * num_dims / 4)
    {
        vstore4((float4) (FLT_MAX), 0, sbest_positions + global_id * 4);
    }
        
    if (global_id < num_swarms / 4)
    {
        vstore4((float4) (FLT_MAX), 0, sbest_fitnesses + global_id * 4);
    }
}
