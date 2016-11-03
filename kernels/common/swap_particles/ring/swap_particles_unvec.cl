/**
 * This kernel performs the actual particle swaps between swarms by shifting values around in
 * global memory. It is executed by one thread per dimension per particle to be swapped
 * total threads: num_swarms * num_exchange * num_tasks / 4
 * Note: if migrating > 50% of the swarm, you need to add global memory barriers here, since bests and worsts will overlap
 */
__kernel void swap_particles_unvec(__global uint *best_pindices,   //buffer holding particle offsets of num_exchange best particles in each swarm (size = num_swarms * num_exchange)
                             __global uint *worst_pindices,  //buffer holding particle offsets of num_exchange best particles in each swarm (size = num_swarms * num_exchange)
                             __global float *position,       //size = num_swarms * num_sparticles * num_tasks
                             __global float *velocity,       //size = num_swarms * num_sparticles * num_tasks
                             __global float *pbest_fitness,  //particle best fitnesses (size = num_swarms * num_sparticles)
                             __global float *pbest_position, //particle best positions (size = num_swarms * num_sparticles * num_tasks)
                             __constant config *conf         //struct containing simulation parameters
    )
{
    uint global_id = get_global_id(0);
    
    uint num_exchange = conf->num_exchange;
    uint num_tasks = conf->num_dims;
    uint num_swarms = conf->num_swarms;
    uint num_sparticles = conf->num_sparticles;

    uint swarm_index = global_id / (num_exchange * num_tasks);
    uint swarm_offset = global_id % (num_exchange * num_tasks);

    if (global_id < num_swarms * num_exchange * num_tasks)
    {
        uint particle_offset = swarm_offset / (num_tasks);
        uint task_offset = swarm_offset % (num_tasks);

        uint worst_position_index = worst_pindices[((swarm_index + 1) % num_swarms) * num_exchange + particle_offset]; //from next swarm
        uint best_position_index = best_pindices[swarm_index * num_exchange + particle_offset];

        uint bposition_offset = (swarm_index * num_sparticles * num_tasks) + (best_position_index) + (task_offset * num_sparticles);

        float temp_pbest_position = pbest_position[bposition_offset];

        float temp_position = position[bposition_offset];

        float temp_velocity = velocity[bposition_offset];

        uint wposition_offset = ((swarm_index + 1) % num_swarms) * num_sparticles * num_tasks + (worst_position_index) + (task_offset * num_sparticles);

        pbest_position[wposition_offset] = temp_pbest_position;//best from this swarm overwrites worst from next swarm
        position[wposition_offset] = temp_position;
        velocity[wposition_offset] = temp_velocity;

        if (swarm_offset < num_exchange)
        {
            //pbest fitnesses
            uint best_fitness_index = best_pindices[swarm_index * num_exchange + swarm_offset];
            uint worst_fitness_index = worst_pindices[((swarm_index + 1) % num_swarms) * num_exchange + swarm_offset];
            float temp_bfitness = pbest_fitness[swarm_index * num_sparticles + best_fitness_index];
            pbest_fitness[((swarm_index + 1) % num_swarms) * num_sparticles + worst_fitness_index] = temp_bfitness;
        }
    }
}
