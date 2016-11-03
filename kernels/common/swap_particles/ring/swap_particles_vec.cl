/**
 * This kernel performs the actual particle swaps between swarms by shifting values around in
 * global memory. It is executed by one thread per dimension per particle to be swapped
 * total threads: num_swarms * num_exchange * num_tasks / 4
 * Note: if migrating > 50% of the swarm, you need to add global memory barriers here, since bests and worsts will overlap
 */
__kernel void swap_particles_vec(
    //__local uint *scratch_best, //size is num_exchange * swarms_per_group
    //__local uint *scratch_worst, //size is num_exchange * swarms_per_group
    __constant uint *best_pindices,   //buffer holding particle offsets of num_exchange best particles in each swarm (size = num_swarms * num_exchange)
    __constant uint *worst_pindices,  //buffer holding particle offsets of num_exchange best particles in each swarm (size = num_swarms * num_exchange)
    //__global uint *best_pindices,
    //__global uint *worst_pindices,
    __global float *position,       //size = num_swarms * num_sparticles * num_tasks
    __global float *velocity,       //size = num_swarms * num_sparticles * num_tasks
    __global float *pbest_fitness,  //particle best fitnesses (size = num_swarms * num_sparticles)
    __global float *pbest_position, //particle best positions (size = num_swarms * num_sparticles * num_tasks)
    uint num_swarms,
    uint num_sparticles,
    uint num_dims,
    uint num_exchange
    )
{
    uint global_id = get_global_id(0);
    uint swarms_per_group = get_local_size(0) / num_exchange;
    uint swarm_id = global_id / num_exchange;
    uint particle_id = global_id % num_exchange;
    
    //if (swarm_id < num_swarms)
    //{
    //uint global_id = get_global_id(0);
        uint next_swarm_id = (swarm_id + 1) % num_swarms;

        uint best_particle_index = best_pindices[swarm_id * num_exchange + particle_id]; //from this swarm
        uint worst_particle_index = worst_pindices[next_swarm_id * num_exchange + particle_id]; //from next swarm

        uint i;
        for (i = 0; i < num_dims / 4; i++)
        {
            float4 this_position = vload4(0, position + swarm_id * num_sparticles * num_dims + best_particle_index * num_dims + i * 4);
            float4 this_velocity = vload4(0, velocity + swarm_id * num_sparticles * num_dims + best_particle_index * num_dims + i * 4);
            float4 this_pbest_pos = vload4(0, pbest_position + swarm_id * num_sparticles * num_dims + best_particle_index * num_dims + i * 4);

            //overwrite worsts in next swarm
            vstore4(this_position, 0, position + next_swarm_id * num_sparticles * num_dims + worst_particle_index * num_dims + i * 4);
            vstore4(this_velocity, 0, velocity + next_swarm_id * num_sparticles * num_dims + worst_particle_index * num_dims + i * 4);
            vstore4(this_pbest_pos, 0, pbest_position + next_swarm_id * num_sparticles * num_dims + worst_particle_index * num_dims + i * 4);
        }

        //pbest fitnesses
        pbest_fitness[next_swarm_id * num_sparticles + worst_particle_index] = pbest_fitness[swarm_id * num_sparticles + best_particle_index];
        //}
}
