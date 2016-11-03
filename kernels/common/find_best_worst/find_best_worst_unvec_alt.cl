/**
 * This kernel finds the indices of the num_exchange best and worst particles in each swarm.
 * It is executed using one thread per particle. Output is written to global memory buffers
 * worst_indices and best_indices.
 * A workgroup size of num_sparticles is used - this means there are num_swarms workgroups.
 */
__kernel void find_best_worst_unvec_alt(
    __global float *fitness,      //size = num_swarms * num_sparticles
    __local float *orig_buf,     //used to hold original fitnesses (in fast local memory) during parallel reduction (size = num_sparticles)
    //since there are num_swarms workgroups, OpenCL gives each workgroup a chunk (accessible as an independent array) with num_sparticles elements
    __local float *worst_buf,    //local buffer used to hold fitnesses during parallel reduction (size = num_sparticles)
    __local uint *worst_indices_buf,
    //since there are num_swarms workgroups, OpenCL gives each workgroup a chunk (accessible as an independent array) with num_sparticles elements
    __local float *best_buf,     //local buffer used to hold fitnesses during parallel reduction (size = num_sparticles)
    __local uint *best_indices_buf,
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
    //uint fitness_offset = group_id * num_sparticles + local_id;
    uint i;
    uint j;
    uint index;
    uint local_mem_offset = (get_local_id(0) / num_sparticles) * num_sparticles * 2;

    //copy the fitnesses of all particles in the swarm from global memory to local buffers
    event_t event = async_work_group_copy(orig_buf,
                                          fitness + get_group_id(0) * num_sparticles * swarms_per_group,
                                          num_sparticles * swarms_per_group,
                                          0
        );
    wait_group_events(1, &event);

    best_buf[local_mem_offset + local_id] = orig_buf[local_mem_offset + local_id];
    worst_buf[local_mem_offset + local_id] = orig_buf[local_mem_offset + local_id];

    best_indices_buf[local_mem_offset + local_id] = local_id;
    worst_indices_buf[local_mem_offset + local_id] = local_id;

    //perform num_exchange parallel reductions, each time finding one worst and one best index
    for (i = 0; i < num_exchange; i++)
    {
        //bring in -1.0s from previous iteration (or init worst and best bufs if 1st iteration)
        worst_buf[local_id + local_mem_offset] = orig_buf[local_id + local_mem_offset];
        best_buf[local_id + local_mem_offset] = orig_buf[local_id + local_mem_offset];

        best_indices_buf[local_mem_offset + local_id] = local_id;
        worst_indices_buf[local_mem_offset + local_id] = local_id;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (j = num_sparticles >> 1; j > 0; j >>= 1)
        {
            if (local_id < j)
            {
                index = local_id + j;

                //negative fitness values indicate that this particle has already been selected (therefore it is
                //better than any other non-selected option left)
                if (best_buf[local_id + local_mem_offset] < 0 ||
                    (best_buf[local_id + local_mem_offset] > best_buf[index + local_mem_offset] && best_buf[index + local_mem_offset] >= 0) )
                {
                    best_buf[local_id + local_mem_offset] = best_buf[index + local_mem_offset];
                    best_indices_buf[local_id + local_mem_offset] = best_indices_buf[index + local_mem_offset];
                }

                //negative fitness values indicate that this particle has already been selected (therefore it is
                //worse than any other non-selected option left)
                if (worst_buf[local_id + local_mem_offset] < 0 ||
                    (worst_buf[local_id + local_mem_offset] <= worst_buf[index + local_mem_offset] && worst_buf[index + local_mem_offset] >= 0) )
                    //note that the above condition is <=, while in the corresponding condition for best_buf (in the previous if statement), it is only > (not >=).
                    //This is necessary to resolve conditions in which we have two identical fitnesses. It ensures that the same fitness does not get picked twice
                    //(as both a best and a worst value).
                {
                    worst_buf[local_id + local_mem_offset] = worst_buf[index + local_mem_offset];
                    worst_indices_buf[local_id + local_mem_offset] = worst_indices_buf[index + local_mem_offset];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (!local_id) //work item 0
        {
            //write the results of the reduction to the output buffers in global memory
            best_indices[group_id * num_exchange + i] = best_indices_buf[0] % num_sparticles;
            worst_indices[group_id * num_exchange + i] = worst_indices_buf[0] % num_sparticles;

            //replace winners' values with -1 in original buffer
            orig_buf[local_mem_offset + (best_indices_buf[0] % num_sparticles)] = -1.0f;
            orig_buf[local_mem_offset + (worst_indices_buf[0] % num_sparticles)] = -1.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
