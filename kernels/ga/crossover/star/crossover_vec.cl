void __kernel crossover_vec(
    __global float *positions, //size = num_swarms * num_sparticles * num_dims
    __global  float *pbest_positions, //size = num_swarms * num_sparticles * num_dims
    __global uint *crossover_perm, //size = num_swarms * num_sparticles
    const uint num_swarms,
    const uint num_sparticles,
    const uint num_dims
    )
{
    uint global_id = get_global_id(0);
    uint abs_particle_id = global_id / (num_dims / 4);
    uint swarm_id = abs_particle_id / num_sparticles;
    uint particle_id = abs_particle_id % num_sparticles;
    uint dim_id = global_id % (num_dims / 4);

    //swarms in [1,num_swarms-1] cross with swarm 0 (result stored in swarms [1, num_swarms-1])
    if (swarm_id > 0)
    {
        //id of the particle we are going to be crossing with from swarm 0 (randomly chosen)
        uint cross_particle_id = crossover_perm[particle_id];

        float4 cur_pbest = vload4(0, pbest_positions + swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4); //from this swarm
        float4 rand_pbest = vload4(0, pbest_positions + cross_particle_id * num_dims + dim_id * 4); //from swarm 0

        //note: there is no need to clamp this value to the solution space here, since we're just taking the midpoint of two existing (valid) points => result will always be in the solution space
        vstore4((cur_pbest + rand_pbest) / (float4) (2.0f), 0, positions + swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4);
    }

    //this is necessary to prevent write-before-read
    barrier(CLK_GLOBAL_MEM_FENCE);

    uint particles_per_swarm = num_sparticles / (num_swarms - 1);
    uint leftovers = num_sparticles % (num_swarms - 1);

    //swarm 0 crosses with a few particles from swarms in [1, num_swarms-1] (result stored in swarm 0)
    if (swarm_id > 0 && particle_id < particles_per_swarm)
    {
        //id of particle from swarms[1, num_swarms-1] that we will cross with
        uint cross_particle_id = crossover_perm[swarm_id * num_sparticles + particle_id];

        float4 cur_pbest = vload4(0, pbest_positions + ((swarm_id - 1) * particles_per_swarm + particle_id) * num_dims + dim_id * 4); //from swarm 0
        float4 rand_pbest = vload4(0, pbest_positions + swarm_id * num_sparticles * num_dims + cross_particle_id * num_dims + dim_id * 4); //from swarms [1, num_swarm-1]

        float4 midpt = (cur_pbest + rand_pbest) / (float4) (2.0f);
        //note: there is no need to clamp this value to the solution space here, since we're just taking the midpoint of two existing (valid) points => result will always be in the solution space
        vstore4(midpt, 0, positions + ((swarm_id - 1) * particles_per_swarm + particle_id) * num_dims + dim_id * 4);

        //printf("s: %u, p: %u, rand: %f, cur: %f, mid: %f\n", swarm_id, particle_id, rand_pbest.x, cur_pbest.x, midpt.x);
    }

    //deal with any leftovers
    if (swarm_id > 0 && swarm_id < leftovers + 1 && particle_id == 0)
    {
        //id of particle from swarms[1, num_swarms-1] that we will cross with
        uint cross_particle_id = crossover_perm[swarm_id * num_sparticles + particles_per_swarm];

        float4 cur_pbest = vload4(0, pbest_positions + ((swarm_id - 1) * particles_per_swarm + particles_per_swarm) * num_dims + dim_id * 4); //from swarm 0
        float4 rand_pbest = vload4(0, pbest_positions + swarm_id * num_sparticles * num_dims + cross_particle_id * num_dims + dim_id * 4); //from swarms [1, num_swarm-1]

        float4 midpt = (cur_pbest + rand_pbest) / (float4) (2.0f);
        //note: there is no need to clamp this value to the solution space here, since we're just taking the midpoint of two existing (valid) points => result will always be in the solution space
        vstore4(midpt, 0, positions + ((num_swarms - 1) * particles_per_swarm + swarm_id - 1) * num_dims + dim_id * 4);

        //printf("s: %u, p: %u, rand: %f, cur: %f, mid: %f\n", swarm_id, particle_id, rand_pbest.x, cur_pbest.x, midpt.x);
    }
}
