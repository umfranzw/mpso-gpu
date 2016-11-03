/* Particle Initialization Kernel */
//one thread per particle dimension vector (total threads = num_swarms * num_sparticles * num_dims).
__kernel void particle_init_unvec(
    __global float *rands,
    __global float *positions,     //size = num_swarms * num_sparticles * num_dims
    __global float *velocities,    //size = num_swarms * num_sparticles * num_dims
    __global float *pbest_positions,
    __global float *pbest_fitnesses,
    __global float *sbest_positions,
    __global float *sbest_fitnesses,
    __constant config *conf        //struct containing simulation parameters
    )
{
    //kernel assumes it is launched with the correct number of threads - one thread for each element.
    uint global_id = get_global_id(0);
    uint max_axis_val = conf->max_axis_val;
    uint num_swarms = conf->num_swarms;
    uint num_sparticles = conf->num_sparticles;
    uint num_dims = conf->num_dims;
    //uint work_items_needed = num_swarms * num_sparticles * num_dims;

    //if (global_id < work_items_needed)
    //{
        //random numbers come in the range [0, 1] - we need to expand the range to [0, num_machines - 1] and [0, num_machines *2] for position and velocity, respectively.
        //each thread will use 2 random numbers - one for setting its particle's position dimension and one for setting its particle's velocity dimension.
        positions[global_id] = rands[global_id] * (max_axis_val - 1);
        velocities[global_id] = rands[get_global_size(0) + global_id] * (max_axis_val / 4.0f);

        pbest_positions[global_id] = FLT_MAX;

        if (global_id < num_swarms * num_sparticles)
        {
            pbest_fitnesses[global_id] = FLT_MAX;
        }

        if (global_id < num_swarms * num_dims)
        {
            sbest_positions[global_id] = FLT_MAX;
        }
        
        if (global_id < num_swarms)
        {
            sbest_fitnesses[global_id] = FLT_MAX;
        }
        //}
}
