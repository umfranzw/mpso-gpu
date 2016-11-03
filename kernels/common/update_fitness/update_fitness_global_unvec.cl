//num_swarms groups of num_sparticles each
__kernel void update_fitness_global_unvec(
    __global float *position,      //size is num_swarms * num_sparticles * num_tasks
    __global float *fitness,       //size is num_swarms * num_sparticles * num_tasks
    __global float *scratch,       //global memory used for scratch - size = num_swarms * num_machines * num_sparticles
    __constant config *conf,     //struct containing simulation parameters
    __constant float *etc_buf,   //ETC matrix (used in makespace computation) size is num_machines * num_tasks
    uint group_size
    )
{
    uint num_sparticles = conf->num_sparticles;
    uint swarms_per_group = group_size / num_sparticles;
    uint local_id = get_local_id(0) % num_sparticles;
    uint group_id = get_group_id(0) * swarms_per_group + (get_local_id(0) / num_sparticles);
    uint num_machines = conf->num_machines;
    uint num_tasks = conf->num_dims;
    //uint num_swarms = conf->num_swarms;

    //if (group_id < num_swarms && local_id < num_sparticles)
    //{
        uint group_pos_base = group_id * (num_sparticles * num_tasks);

        //zero out the scratch array
        uint i;
        for (i = 0; i < num_machines; i++)
        {
            scratch[group_id * num_sparticles * num_machines + local_id * num_machines + i] = 0.0f;
        }

        float makespan = -1.0f;
        float position_el;
        uint machine_el;
        uint etc_offset;
        uint scratch_offset;
        float etc_val;
        float scratch_val;

        for (i = 0; i < num_tasks; i++)
        {
            position_el = position[group_pos_base + local_id + i * num_sparticles];
            machine_el = (uint) (position_el + 0.5f);
            etc_offset = machine_el * num_tasks + i;
            scratch_offset = group_id * num_sparticles * num_machines + local_id * num_machines + machine_el;

            etc_val = etc_buf[etc_offset];

            //this must be done sequentially to allow combining if two vector elements refer to the same machine
            scratch[scratch_offset] += etc_val;

            scratch_val = scratch[scratch_offset];
            makespan = fmax(makespan, scratch_val);
        }

        //is this really necessary?
        barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to sync up to gain benefit of global memory coalescing for final write to fitness array
        fitness[group_id * num_sparticles + local_id] = makespan;
        //}
}
