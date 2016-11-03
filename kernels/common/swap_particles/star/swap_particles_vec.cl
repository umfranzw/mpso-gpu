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
    __global float *positions,       //size = num_swarms * num_sparticles * num_tasks
    __global float *velocities,       //size = num_swarms * num_sparticles * num_tasks
    __global float *pbest_fitnesses,  //particle best fitnesses (size = num_swarms * num_sparticles)
    __global float *pbest_positions, //particle best positions (size = num_swarms * num_sparticles * num_tasks)
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
    
    uint write_particles_per_swarm = num_exchange / (num_swarms - 1); //will be 0 if num_exchange < num_swarms
    uint leftover_write_particles = num_exchange % (num_swarms - 1); //if the above was 0, this will be > 0

    //grab best data from swarm 0 and overwrite this swarm's worst data
    if (swarm_id > 0 && swarm_id < num_swarms) //guarentee: particle_id < num_exchange
    {
        uint worst_particle_index = worst_pindices[swarm_id * num_exchange + particle_id]; //from this swarm
        uint best_particle_index = best_pindices[particle_id]; //from swarm 0

        uint i;
        for (i = 0; i < num_dims / 4; i++)
        {
            float4 best_pos = vload4(0, positions + best_particle_index * num_dims + i * 4); //load from swarm 0
            float4 best_vel = vload4(0, velocities + best_particle_index * num_dims + i * 4); //load from swarm 0
            float4 pbest_pos = vload4(0, pbest_positions + best_particle_index * num_dims + i * 4); //load from swarm 0

            vstore4(best_pos, 0, positions + swarm_id * num_sparticles * num_dims + worst_particle_index * num_dims + i * 4); //store to this swarm
            vstore4(best_vel, 0, velocities + swarm_id * num_sparticles * num_dims + worst_particle_index * num_dims + i * 4); //store to this swarm
            vstore4(pbest_pos, 0, pbest_positions + swarm_id * num_sparticles * num_dims + worst_particle_index * num_dims + i * 4); //store to this swarm
        }

        //pbest_fitnesses - grab best data from swarm 0 and overwrite this swarm's worst data
        pbest_fitnesses[swarm_id * num_sparticles + worst_particle_index] = pbest_fitnesses[best_particle_index];
    }

    //grab best data from this swarm and overwrite swarm 0's worst data
    if (swarm_id > 0 && swarm_id < num_swarms && particle_id < write_particles_per_swarm) //deal with regular case
    {
        uint worst_particle_index = worst_pindices[particle_id + swarm_id - 1]; //from swarm 0
        uint best_particle_index = best_pindices[swarm_id * num_exchange + particle_id]; //from this swarm

        uint i;
        for (i = 0; i < num_dims / 4; i++)
        {
            float4 best_pos = vload4(0, positions + swarm_id * num_sparticles * num_dims + best_particle_index * num_dims + i * 4); //load from this swarm
            float4 best_vel = vload4(0, velocities + swarm_id * num_sparticles * num_dims + best_particle_index * num_dims + i * 4); //load from this swarm
            float4 pbest_pos = vload4(0, pbest_positions + swarm_id * num_sparticles * num_dims + best_particle_index * num_dims + i * 4); //load from this swarm

            vstore4(best_pos, 0, positions + worst_particle_index * num_dims + i * 4); //store to swarm 0
            vstore4(best_vel, 0, velocities + worst_particle_index * num_dims + i * 4); //store to swarm 0
            vstore4(pbest_pos, 0, pbest_positions + worst_particle_index * num_dims + i * 4); //store to swarm 0
        }

        //pbest_fitnesses - grab best data from this swarm and overwrite swarm 0's worst data
        pbest_fitnesses[worst_particle_index] = pbest_fitnesses[swarm_id * num_sparticles + best_particle_index];
    }

    //one thread per swarm deals with the leftovers, if necessary
    if (swarm_id > 0 && swarm_id < leftover_write_particles + 1 && !particle_id)
    {
        uint worst_particle_index = worst_pindices[write_particles_per_swarm * (num_swarms - 1) + swarm_id - 1]; //from swarm 0
        uint best_particle_index = best_pindices[swarm_id * num_exchange + write_particles_per_swarm]; //from this swarm

        uint i;
        for (i = 0; i < num_dims / 4; i++)
        {
            float4 best_pos = vload4(0, positions + swarm_id * num_sparticles * num_dims + best_particle_index * num_dims + i * 4); //load from this swarm
            float4 best_vel = vload4(0, velocities + swarm_id * num_sparticles * num_dims + best_particle_index * num_dims + i * 4); //load from this swarm
            float4 pbest_pos = vload4(0, pbest_positions + swarm_id * num_sparticles * num_dims + best_particle_index * num_dims + i * 4); //load from this swarm

            vstore4(best_pos, 0, positions + worst_particle_index * num_dims + i * 4); //store to swarm 0
            vstore4(best_vel, 0, velocities + worst_particle_index * num_dims + i * 4); //store to swarm 0
            vstore4(pbest_pos, 0, pbest_positions + worst_particle_index * num_dims + i * 4); //store to swarm 0
        }

        //pbest_fitnesses - grab best data from this swarm and overwrite swarm 0's worst data
        pbest_fitnesses[worst_particle_index] = pbest_fitnesses[swarm_id * num_sparticles + best_particle_index];
    }
}
