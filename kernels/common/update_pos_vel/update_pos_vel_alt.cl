//num_swarms groups of size num_sparticles
__kernel void update_pos_vel_alt(__global float *positions,               //size is num_swarms * num_sparticles * num_tasks
                             __global float *velocities,              //size is num_swarms * num_sparticles * num_tasks
                             __global float *particle_best_positions, //size is num_swarms * num_sparticles * num_tasks
                             __constant float *swarm_best_positions,    //size is num_swarms * num_tasks
                             __global float *rands,                   //buffer of random numbers of size MAX_NUM_RANDS
                             uint rands_offset,             //index at which to start accessing rands
                             //uint iter_index,
                             uint group_size,
                             __constant config *conf                //struct containing simulation parameters
    )
{
    uint local_id = get_local_id(0);
    
    if (local_id < group_size)
    {
        uint global_id = get_global_id(0);
        uint group_id = get_group_id(0);
        uint num_groups = get_num_groups(0);
        
        uint num_swarms = conf->num_swarms;
        uint num_sparticles = conf->num_sparticles;
        uint num_tasks = conf->num_dims;
        uint num_machines = conf->num_machines;

        uint swarms_per_group = group_size / num_sparticles;

        float4 pos_chunk;
        float4 best_chunk;
        float4 rands_chunk;
        float4 vel_chunk;

        uint i;
        //uint task_offset;
        //uint swarm_offset;
        for (i = 0; i < (num_tasks / 4) * swarms_per_group; i++)
        {
            /* task_offset = i / swarms_per_group; */
            /* swarm_offset = i % swarms_per_group; */
            
            pos_chunk = vload4(0, positions + (group_id + (i % swarms_per_group)) * num_sparticles * num_tasks + local_id * 4 + (i / swarms_per_group) * 4 * num_sparticles);
            best_chunk = vload4(0, particle_best_positions + (group_id + (i % swarms_per_group)) * num_sparticles * num_tasks + local_id * 4 + (i / swarms_per_group) * 4 * num_sparticles);
            vel_chunk = vload4(0, velocities + (group_id + (i % swarms_per_group)) * num_sparticles * num_tasks + local_id * 4 + (i / swarms_per_group) * 4 * num_sparticles);
            rands_chunk = vload4(0, rands + rands_offset + local_id * group_id * (group_size * swarms_per_group) * 4 + i * num_groups * (group_size * swarms_per_group) * 4);

            vel_chunk *= conf->omega;
            vel_chunk += conf->c1 * rands_chunk * (best_chunk - pos_chunk);

            best_chunk = vload4(0, swarm_best_positions + (group_id + (i % swarms_per_group)) + (i / swarms_per_group) * num_swarms * 4);
            rands_chunk = vload4(0, rands + rands_offset + local_id * group_id * (group_size * swarms_per_group) * 4 + i * num_groups * (group_size * swarms_per_group) * 4);
            vel_chunk += conf->c2 * rands_chunk * (best_chunk - pos_chunk);

            //clamp velocity to [-machines / 2, num_machines / 2]
            vel_chunk = clamp( vel_chunk, (float4) (num_machines * -0.5f), (float4) (num_machines * 0.5f) );
            vstore4(vel_chunk, 0, velocities + (group_id + (i % swarms_per_group)) * num_sparticles * num_tasks + local_id * 4 + (i / swarms_per_group) * 4 * num_sparticles);
        
            //update position based on the above velocity.
            //Clamp position to solution space - clamp() function is from OpenCL API
            pos_chunk = clamp( pos_chunk + vel_chunk, (float4) 0.0f, (float4) (num_machines - 1) );
            vstore4(pos_chunk, 0, positions + (group_id + (i % swarms_per_group)) * num_sparticles * num_tasks + local_id * 4 + (i / swarms_per_group) * 4 * num_sparticles);
        }
        
        /* for (i = 0; i < num_tasks / 4; i++) */
        /* { */
        /*     for (j = 0; j < swarms_per_group; j++) */
        /*     { */
                
        /*     } */
        /* } */
    }

    //update velocity (using standard PSO equation)
    /* sbest_chunk = conf->omega * vel_chunk + */
    /*     conf->c1 * rands_chunk1 * */
    /*     (pbest_chunk - pos_chunk) + */
    /*     conf->c2 * rands_chunk2 * */
    /*     (sbest_chunk - pos_chunk); */    
}
