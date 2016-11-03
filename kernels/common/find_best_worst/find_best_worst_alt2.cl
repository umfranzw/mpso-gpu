//launch with num_swarms groups of num_sparticles / 2 threads
__kernel void find_best_worst_alt2(
    __global float *fitness,
    __local float *scratch, //size per workgroup is num_sparticles
    __global uint *worst_indices,
    __global uint *best_indices,
    uint num_swarms,
    uint num_sparticles,
    uint num_dims,
    uint num_exchange
    )
{
    uint swarms_per_group = get_local_size(0) / (num_sparticles / 2);
    uint local_id = get_local_id(0) % (num_sparticles / 2);
    uint group_id = get_group_id(0) * swarms_per_group + get_local_id(0) / (num_sparticles / 2);
    uint local_mem_offset = (get_local_id(0) / (num_sparticles / 2)) * num_sparticles; //start of this thread's group's chunk of memory

    //copy fitnesses to local memory so we don't have to load float2 from global mem with each thread (only uses half bandwidth)
    //this is not beneficial, since these values are never used more than once
    /* event_t event = async_work_group_copy(scratch + get_group_id(0) * num_sparticles * swarms_per_group, */
    /*                                       fitness + get_group_id(0) * num_sparticles * swarms_per_group, */
    /*                                       num_sparticles * swarms_per_group, */
    /*                                       0 */
    /*     ); */
    /* wait_group_events(1, &event); */

    //copy fitnesses to local memory
    float2 fitness_chunk;
    int2 counts = (int2) (0); //win count
    int2 minus_one = (int2) (-1);
    float2 orig_chunk;

    //if (group_id < num_swarms)
    //{
        orig_chunk = vload2(0, fitness + group_id * num_sparticles + local_id * 2);
        vstore2(orig_chunk, 0, scratch + local_mem_offset + local_id * 2);

        //compare to own vec
        counts += (minus_one * (orig_chunk > orig_chunk.yx));
        //}
    barrier(CLK_LOCAL_MEM_FENCE);

    //if (group_id < num_swarms)
    //{
        //compare to other vecs
        uint i;
        uint next_index;
        for (i = 1; i < num_sparticles / 2; i++)
        {
            next_index = ((local_id + i) * 2) % (num_sparticles);
            fitness_chunk = vload2(0, scratch + local_mem_offset + next_index);

            counts += (minus_one * (orig_chunk > fitness_chunk.xy));
            counts += (minus_one * (orig_chunk > fitness_chunk.yx));
        }
        //}
    barrier(CLK_LOCAL_MEM_FENCE);

    //some elements may be identical, and therefore we may have more than one thread with the same count. We need to deal with this.
    
    //int4 indices = ((int4) (local_id * 4)) + ((int4) (0, 1, 2, 3)); //this increases register count by one
    int2 indices = (int2) (local_id * 2, local_id * 2 + 1);
    int2 test_chunk;
    int2 cmp = minus_one;
    int done = 0;

    //note: barriers here will sync up all swarms in this workgroup!
    do
    {
        //note: these writes may conflict
        if (!done)// && group_id < num_swarms) //nesting this results in less register usage than applying the !done condition into each individual if statement
        {
            if (cmp.x)
            {
                scratch[local_mem_offset + counts.x] = (float) indices.x;

                /* atomic_xchg( */
                /*     scratch + local_mem_offset + counts.x, */
                /*     (float) indices.x */
                /*     ); */
            }
            if (cmp.y)
            {
                scratch[local_mem_offset + counts.y] = (float) indices.y;

                /* atomic_xchg( */
                /*     scratch + local_mem_offset + counts.y, */
                /*     (float) indices.y */
                /*     ); */
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //read them back to see if we had any write conflicts
        if (!done)// && group_id < num_swarms)
        {
            //only read back if we wrote to the location, above
            test_chunk.x = (cmp.x ? (int) scratch[local_mem_offset + counts.x] : indices.x);
            test_chunk.y = (cmp.y ? (int) scratch[local_mem_offset + counts.y] : indices.y);
            
            cmp = (test_chunk != indices); //sets vector components to -1 where test_chunk != indices. Sets components to 0 otherwise.
            done = !any(cmp);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //reset flag
        if (!get_local_id(0)) //no  && group_id < num_swarms here! We want the loop to terminate!
        {
            scratch[0] = 1.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!done)// && group_id < num_swarms) //if any component is set to -1
        {
            counts += (minus_one * cmp); //invert cmp to get positive 1s, then add to counts. This increments each component of counts that wasn't sucessfully written to local memory (due to a write conflict in which this thread's value was overwritten).
            //set flag to indicate that another iteration is needed
            
            scratch[0] = -1.0f; //note: if this is not hit, scratch[local_mem_offset] is guarenteed to be >= 0, since all fitnesses and counts (the two previous things that have been copied to local mem) are >= 0.
            /* atomic_xchg( */
            /*     scratch, */
            /*     -1.0f */
            /*     ); */
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } while (scratch[0] < 0.0f);

    //update the global memory arrays
    //At this point, counts will have been updated such that collectively, all the threads registers contain the values in [0, num_sparticles], indicating how many wins each element had. Conflicts where two elements were identical have been worked out by the above loop.
    //bests
    //if (group_id < num_swarms)
    //{
        cmp = counts < (int2) (num_exchange);
        test_chunk = (group_id * num_exchange) + counts;
        if (cmp.x)
        {
            best_indices[test_chunk.x] = indices.x;
        }
        if (cmp.y)
        {
            best_indices[test_chunk.y] = indices.y;
        }

        //worsts
        cmp = counts >= (int2) (num_sparticles - num_exchange);
        test_chunk = (group_id * num_exchange + (num_sparticles - 1)) - counts;
        if (cmp.x)
        {
            worst_indices[test_chunk.x] = indices.x;
        }
        if (cmp.y)
        {
            worst_indices[test_chunk.y] = indices.y;
        }
        //}
}
