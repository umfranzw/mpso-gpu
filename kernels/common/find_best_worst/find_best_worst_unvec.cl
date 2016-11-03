/**
 * This kernel finds the indices of the num_exchange best and worst particles in each swarm.
 * It is executed using one thread per particle. Output is written to global memory buffers
 * worst_indices and best_indices.
 * A workgroup size of num_sparticles is used - this means there are num_swarms workgroups.
 */
__kernel void find_best_worst_unvec(
    __global float *fitness,      //size = num_swarms * num_sparticles
    __local float2 *orig_buf,     //used to hold original fitnesses (in fast local memory) during parallel reduction (size = num_sparticles)
    //since there are num_swarms workgroups, OpenCL gives each workgroup a chunk (accessible as an independent array) with num_sparticles elements
    __local float2 *worst_buf,    //local buffer used to hold fitnesses during parallel reduction (size = num_sparticles)
    //since there are num_swarms workgroups, OpenCL gives each workgroup a chunk (accessible as an independent array) with num_sparticles elements
    __local float2 *best_buf,     //local buffer used to hold fitnesses during parallel reduction (size = num_sparticles)
    //since there are num_swarms workgroups, OpenCL gives each workgroup a chunk (accessible as an independent array) with num_sparticles elements
    __global uint *worst_indices, //output buffer of size num_swarms * num_exchange
    __global uint *best_indices,  //output buffer of size num_swarms * num_exchange
    __constant config *conf     //struct containing simulation parameters
    )
{
    uint num_sparticles = conf->num_sparticles;
    uint swarms_per_group = get_local_size(0) / num_sparticles;
    uint local_id = get_local_id(0) % num_sparticles; //corresponds to offset within swarm
    uint group_id = get_group_id(0) * swarms_per_group + get_local_id(0) / num_sparticles; //each id corresponds to one swarm (there are run->num_swarms workgroups in total)
    //assign some frequently used struct params to local registers so we don't have to keep accessing (uncached) global memory
    uint num_exchange = conf->num_exchange;
    uint fitness_offset = group_id * num_sparticles + local_id;
    uint i;
    uint j;
    uint index;
    uint local_mem_offset = (get_local_id(0) / num_sparticles) * num_sparticles; //* 2 is implicit, since we are addressing float2 arrays
    uint num_swarms = conf->num_swarms;

    if (group_id < num_swarms)
    {
    //copy the fitnesses of all particles in the swarm from global memory to local buffers
    //note: we track fitness and fitness index using a handy two-tuple
    orig_buf[local_id + local_mem_offset].x = fitness[fitness_offset];
    orig_buf[local_id + local_mem_offset].y = fitness_offset;
    }

    //perform num_exchange parallel reductions, each time finding one worst and one best index
    for (i = 0; i < num_exchange; i++)
    {
        if (group_id < num_swarms)
        {
        //bring in -1.0s from previous iteration (or init worst and best bufs if 1st iteration)
        worst_buf[local_id + local_mem_offset] = orig_buf[local_id + local_mem_offset];
        best_buf[local_id + local_mem_offset] = orig_buf[local_id + local_mem_offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (j = num_sparticles >> 1; j > 0; j >>= 1)
        {
            if (local_id < j && group_id < num_swarms)
            {
                index = local_id + j;

                //negative fitness values indicate that this particle has already been selected (therefore it is
                //better than any other non-selected option left)
                if (best_buf[local_id + local_mem_offset].x < 0 ||
                    (best_buf[local_id + local_mem_offset].x > best_buf[index + local_mem_offset].x && best_buf[index + local_mem_offset].x >= 0) )
                {
                    best_buf[local_id + local_mem_offset] = best_buf[index + local_mem_offset];
                }

                //negative fitness values indicate that this particle has already been selected (therefore it is
                //worse than any other non-selected option left)
                if (worst_buf[local_id + local_mem_offset].x < 0 ||
                    (worst_buf[local_id + local_mem_offset].x <= worst_buf[index + local_mem_offset].x && worst_buf[index + local_mem_offset].x >= 0) )
                    //note that the above condition is <=, while in the corresponding condition for best_buf (in the previous if statement), it is only > (not >=).
                    //This is necessary to resolve conditions in which we have two identical fitnesses. It ensures that the same fitness does not get picked twice
                    //(as both a best and a worst value).
                {
                    worst_buf[local_id + local_mem_offset] = worst_buf[index + local_mem_offset];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (!local_id && group_id < num_swarms) //work item 0
        {
            //write the results of the reduction to the output buffers in global memory
            best_indices[group_id * num_exchange + i] = ((uint) best_buf[0].y) % num_sparticles;
            worst_indices[group_id * num_exchange + i] = ((uint) worst_buf[0].y) % num_sparticles;

            //replace winners' values with -1 in original buffer
            orig_buf[((uint) best_buf[0].y) - (group_id * num_sparticles) + local_mem_offset].x = -1.0f;
            orig_buf[((uint) worst_buf[0].y) - (group_id * num_sparticles) + local_mem_offset].x = -1.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
