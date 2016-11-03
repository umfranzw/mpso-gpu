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

    //cross with particle from next swarm
    uint cross_particle_id = crossover_perm[((swarm_id + 1) % num_swarms) * num_sparticles + particle_id];

    float4 cur_pbest = vload4(0, pbest_positions + swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4); //from this swarm
    float4 rand_pbest = vload4(0, pbest_positions + swarm_id * num_sparticles * num_dims + cross_particle_id * num_dims + dim_id * 4); //from next swarm

    //note: there is no need to clamp this value to the solution space here, since we're just taking the midpoint of two existing (valid) points => result will always be in the solution space
    vstore4((cur_pbest + rand_pbest) / (float4) (2.0f), 0, positions + swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4);
}
