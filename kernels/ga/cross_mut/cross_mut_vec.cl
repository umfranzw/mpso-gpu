void __kernel cross_mut_vec(
    __global float *positions, //size = num_swarms * num_sparticles * num_dims
    __constant uint *crossover_perm, //size = num_swarms * num_sparticles
    const uint num_swarms,
    const uint num_sparticles,
    const uint num_dims,
    const float max_axis_val
    )
{
    uint global_id = get_global_id(0);
    uint abs_particle_id = global_id / (num_dims / 4);
    uint swarm_id = abs_particle_id / num_sparticles;
    uint particle_id = abs_particle_id % num_sparticles;
    uint dim_id = global_id % (num_dims / 4);

    uint cross_particle_id = crossover_perm[abs_particle_id];

    float4 rands;

    //decision to crossover is performed on a per particle basis
    //all threads operating on this particle will generate the same vector of rands, then pick the appropriate one - this way we can avoid using local memory to share the value
    float4 float_rands = get_float_rands_vec(
        abs_particle_id,
        CROSSOVER_STREAM,
        iter_index,
        seed
        );
    uint4 shuffle_mask = ((uint4) (abs_particle_id % 4) + (uint4) (0, 3, 2, 1)) % (uint4) 4;
    //here's how the line above works:
    //abs_particle_id % 4     shuffle_mask
    // 0                              (0, 3, 2, 1)
    // 1                              (1, 0, 3, 2)
    // 2                              (2, 1, 0, 3)
    // 3                              (3, 2, 1, 0)
    
    float_rands = shuffle(float_rands, shuffle_mask);
    //now this thread's random number is at float_rands.x
    
    if (float_rands.x < cross_prob)
    {
        float4 parent1_chunk = vload4(0, positions + swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4); //current position (from this swarm)
        float4 parent2_chunk = vload4(0, positions + swarm_id * num_sparticles * num_dims + cross_particle_id * num_dims + dim_id * 4); //random position (from this swarm)
        float4 child1_chunk;
        float4 child2_chunk;
        
        //do crossover
        uint4 uint_rands = get_uint_rands_vec(
            num_swarms * num_sparticles + abs_particle_id,
            CROSSOVER_STREAM,
            iter_index,
            seed
        ) % num_dims;
        uint4 shuffle_mask = ((uint4) (abs_particle_id % 4) + (uint4) (0, 3, 2, 1)) % (uint4) 4;
        uint_rands = shuffle(uint_rands, shuffle_mask);
        uint cross_pt = uint_rands.x;

        intersect_index = cross_pt / 4;
        intersecting = cross_pt % 4;

        //chromosome example (cross_pt = 2):
        //parent1: A0 A1 A2 | A3 A4 A5 A6 A7
        //parent2: B0 B1 B2 | B3 B4 B5 B6 B7
        //child1  : A0 A1 A2 | B3 B4 B5 B6 B7
        //child2  : B0 B1 B2 | A3 A4 A5 A6 A7
        
        //non-intersecting chunks
        child1_chunk = (dim_id < intersect_index) ? parent1_chunk : parent2_chunk;
        child2_chunk = (dim_id < intersent_index) ? parent2_chunk : parent1_chunk;

        //intersecting chunk (if any)
        if (intersecting && dim_id == intersect_index)
        {
            int4 cmp = (uint4) (0, 1, 2, 3) < (uint4) intersecting;
            child1_chunk = select(parent2_chunk, parent1_chunk, cmp);
            child2_chunk = select(parent2_chunk, parent1_chunk, !cmp);
        }
    }
    barrier(GLOBAL_MEM_FENCE);
    
    if (rands.x < cross_prob)
    {
        //overwrite parents
        vstore4(child1_chunk, 0, positions + swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4);
        vstore4(child2_chunk, 0, positions + swarm_id * num_sparticles * num_dims + cross_particle_id * num_dims + dim_id * 4);
    }
    barrier(GLOBAL_MEM_FENCE);

    //decision to mutate is performed on a per dimension basis
    float_rands = get_float_rands_vec(
        abs_particle_id * num_dims + dim_id,
        MUTATION_STREAM,
        iter_index,
        seed
        );

    int4 cmp = float_rands < mut_prob;

    float_rands = get_float_rands_vec(
        num_swarms * num_sparticles * num_dims + abs_particle_id * num_dims + dim_id,
        MUTATION_STREAM,
        iter_index,
        seed
        );

    chunk = vload4(0, positions + swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4);
    chunk = select(chunk, chunk * float_rands, cmp);
    vstore4(chunk, swarm_id * num_sparticles * num_dims + particle_id * num_dims + dim_id * 4);
}
