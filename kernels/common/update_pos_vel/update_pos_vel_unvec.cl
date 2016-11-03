/**
 * This kernel handles updating the position and velocity global memory buffers. It uses the
 * standard PSO equations described in the paper.
 * This kernel is executed using one thread per particle dimension (total threads = num_swarms * num_sparticles * num_tasks).
 */
__kernel void update_pos_vel_unvec(__global float *positions,               //size is num_swarms * num_sparticles * num_tasks
                             __global float *velocities,              //size is num_swarms * num_sparticles * num_tasks
                             __global float *particle_best_positions, //size is num_swarms * num_sparticles * num_tasks
                             __global float *swarm_best_positions,    //size is num_swarms * num_tasks
                             __global float *rands,                   //buffer of random numbers of size MAX_NUM_RANDS
                             uint rands_offset,             //index at which to start accessing rands
                             __constant config *conf                //struct containing simulation parameters
    )
{
    uint global_id = get_global_id(0);
    uint num_swarms = conf->num_swarms;
    uint num_sparticles = conf->num_sparticles;
    uint num_tasks = conf->num_dims;
    uint num_machines = conf->num_machines;

    //if (global_id < num_swarms * num_sparticles * num_tasks)
    //{
        uint swarm_index = global_id / (num_tasks * num_sparticles);
        uint task_index = (global_id / num_sparticles) % num_tasks;
        uint r_offset = rands_offset + global_id * 2;
    
        //update velocity (using standard PSO equation)
        float new_vel = conf->omega * velocities[global_id] +
            conf->c1 * rands[r_offset] *
            (particle_best_positions[global_id] - positions[global_id]) +
            conf->c2 * rands[r_offset + 1] *
            (swarm_best_positions[swarm_index + task_index * num_swarms] - positions[global_id]);

        //clamp velocity to [-machines / 2, num_machines / 2]
        new_vel = clamp(new_vel, ((float) num_machines) * -0.5f, ((float) num_machines) * 0.5f);
        velocities[global_id] = new_vel;
        
        //update position based on the above velocity.
        //Clamp position to solution space - clamp() function is from OpenCL API
        positions[global_id] = clamp(positions[global_id] + new_vel, 0.0f, (float) (num_machines - 1));
        //}
}
