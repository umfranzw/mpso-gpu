/*
 Particle Initialization Kernel
 Global size is one thread per particle dimension vector (total threads = num_swarms * num_sparticles * num_dims / 4).
*/

__kernel void particle_init_vec(
    __global float *positions,       //size = num_swarms * num_sparticles * num_dims
    __global float *velocities,      //size = num_swarms * num_sparticles * num_dims
    __global float *pbest_positions, //size = num_swarms * num_sparticles * num_dims
    __global float *pbest_fitnesses, //size = num_swarms * num_sparticles
    __global float *sbest_positions, //size = num_swarms * num_dims
    __global float *sbest_fitnesses, //size = sum_swarms
    __global float *mut_counts,  //size = num_swarms
    __global uint *swarm_health,
    uint rep,
    uint num_swarms,
    uint num_sparticles,
    float max_axis_val,
    uint seed,
    float max_vel,
    uint num_reps,
    __global uint *alg_health
    )
{
    //kernel assumes it is launched with the correct number of threads - one thread for each element.
    uint global_id = get_global_id(0);

    //random numbers come in the range [0, 1] - we need to expand the range to [-max_axis_val, max_axis_val] and [-max_axis_val / 4, max_axis_val / 4] for position and velocity, respectively.
    //each thread will use 2 random numbers - one for setting its particle's position dimension and one for setting its particle's velocity dimension.
    //Here, we use 2 vec4s of rands, since each thread is updating 4 dimensions simultaneously.
        
    float4 r1 = get_float_rands_vec(
        global_id,
        PARTICLE_INIT_STREAM,
        0,
        seed
        );

    vstore4(r1 * max_axis_val * 2 - max_axis_val, 0, positions + global_id * 4);
    vstore4((float4) 0.0f, 0, velocities + global_id * 4);

    /* float4 r2 = get_float_rands_vec( */
    /*     global_id + get_global_size(0), */
    /*     PARTICLE_INIT_STREAM, */
    /*     0, */
    /*     seed */
    /*     ); */
    /* vstore4(r2 * max_axis_val * 2 - max_axis_val, 0, velocities + global_id * 4); */

    if (global_id < num_swarms * num_sparticles / 4)
    {
        vstore4((float4) (FLT_MAX), 0, pbest_fitnesses + global_id * 4);
    }

    if (global_id < num_swarms / 4)
    {
        vstore4((float4) (FLT_MAX), 0, sbest_fitnesses + global_id * 4);
        vstore4((uint4) (0), 0, swarm_health + global_id * 4);
        vstore4((uint4) (0), 0, alg_health + global_id * 4);
    }

    uint i;
    for (i = global_id; !rep && i < (num_reps * num_swarms) / 4; i += get_global_size(0))
    {
        //vstore4((float4) (0), 0, mut_counts + global_id * 4 + i * num_swarms + num_reps); //num_reps offset for GA samples
        vstore4((float4) (0), 0, mut_counts + i * 4 + num_reps); //num_reps offset for GA samples
    }
}
