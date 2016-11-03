//num_swarms groups of num_sparticles each
__kernel void update_fitness_shared_unvec(
    __global float *position,      //size is num_swarms * num_sparticles * num_tasks
    __global float *fitness,       //size is num_swarms * num_sparticles * num_tasks
    __local float *scratch,       //local memory used for scratch - size = num_machines * num_sparticles
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

    uint local_mem_offset = (get_local_id(0) / num_sparticles) * num_sparticles * num_machines;
    uint group_pos_base = group_id * (num_sparticles * num_tasks);

    //zero out the scratch array
    uint i;
    for (i = 0; i < num_machines; i++)
    {
        scratch[local_mem_offset + local_id * num_machines + i] = 0.0f;
    }

    float makespan = -1.0f;
    float position_el;
    uint machine_el;
    uint scratch_offset;
    float temp;

    for (i = 0; i < num_tasks; i++)
    {
        position_el = position[group_pos_base + local_id + i * num_sparticles];
        
        machine_el = (uint) (position_el + 0.5f);
        scratch_offset = local_mem_offset + local_id * num_machines + machine_el;

        temp = scratch[scratch_offset];
        temp += etc_buf[machine_el * num_tasks + i];
        scratch[scratch_offset] = temp;

        makespan = fmax(makespan, temp);
    }

    fitness[group_id * num_sparticles + local_id] = makespan;
}
