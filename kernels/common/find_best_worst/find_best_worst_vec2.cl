//Doesn't make sense to group this one (multiple swarms per group), since that uses so much local memory that it reduces the number of active wavefronts.

/**
 * This kernel finds the indices of the num_exchange best and worst particles in each swarm.
 * It is executed using one thread per particle. Output is written to global memory buffers
 * worst_indices and best_indices.
 * A workgroup size of num_sparticles is used - this means there are num_swarms workgroups.
 */
__kernel void find_best_worst_vec2(
    __global float *fitness,      //size = num_swarms * num_sparticles
    __local float *orig_buf,     //used to hold original fitnesses (in fast local memory) during parallel reduction (size = num_sparticles)
    //since there are num_swarms workgroups, OpenCL gives each workgroup a chunk (accessible as an independent array) with num_sparticles elements
    __local float *worst_buf,    //local buffer used to hold fitnesses during parallel reduction (size = num_sparticles)
    //since there are num_swarms workgroups, OpenCL gives each workgroup a chunk (accessible as an independent array) with num_sparticles elements
    __local float *best_buf,     //local buffer used to hold fitnesses during parallel reduction (size = num_sparticles)
    //since there are num_swarms workgroups, OpenCL gives each workgroup a chunk (accessible as an independent array) with num_sparticles elements
    __global uint *worst_indices, //output buffer of size num_swarms * num_exchange
    __global uint *best_indices,  //output buffer of size num_swarms * num_exchange
    uint num_swarms,
    uint num_sparticles,
    uint num_dims,
    uint num_exchange
    )
{
    uint swarms_per_group = get_local_size(0) / (num_sparticles / 2);
    uint local_id = get_local_id(0) % (num_sparticles / 2); //corresponds to offset within swarm
    uint group_id = get_group_id(0) * swarms_per_group + get_local_id(0) / (num_sparticles / 2); //each id corresponds to one swarm (there are run->num_swarms workgroups in total)
    //assign some frequently used struct params to local registers so we don't have to keep accessing (uncached) global memory
    uint fitness_offset = group_id * num_sparticles + local_id * 2;
    uint i;
    uint j;
    uint index;
    uint local_mem_offset = ( get_local_id(0) / (num_sparticles / 2) ) * num_sparticles * 2;

    const float2 zero = (float2) (0.0f, 0.0f);

    if (group_id < num_swarms)
    {
        //copy the fitnesses of all particles in the swarm from global memory to local buffers
        //note: we track fitness and fitness index using a handy two-tuple
        float2 fitness_vec = vload2(0, fitness + fitness_offset);
        float4 info_vec = (float4) (fitness_vec.x, fitness_vec.y, fitness_offset, fitness_offset + 1);
        vstore4(info_vec, 0, orig_buf + local_mem_offset + local_id * 4);

        vstore2(info_vec.zw, 0, best_buf + local_mem_offset + local_id * 4 + 2);
        vstore2(info_vec.zw, 0, worst_buf + local_mem_offset + local_id * 4 + 2);
    }

    float4 left_vec;
    float4 right_vec;
    int2 cmp;
    
    //perform num_exchange parallel reductions, each time finding one worst and one best index
    for (i = 0; i < num_exchange; i++)
    {
        if (group_id < num_swarms)
        {
        //bring in -1.0s from previous iteration (or init worst and best bufs if 1st iteration)
        vstore4(vload4(0, orig_buf + local_mem_offset + local_id * 4), 0, worst_buf + local_mem_offset + local_id * 4);
        vstore4(vload4(0, orig_buf + local_mem_offset + local_id * 4), 0, best_buf + local_mem_offset + local_id * 4);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        for (j = num_sparticles >> 2; j > 0; j >>= 1)
        {
            if (local_id < j && group_id < num_swarms)
            {
                index = local_id + j;

                //negative fitness values indicate that this particle has already been selected (therefore it is
                //better than any other non-selected option left)
                left_vec = vload4(0, best_buf + local_mem_offset + local_id * 4); //only load the first 2 elements (the fitnesses)
                right_vec = vload4(0, best_buf + local_mem_offset + index * 4);
                cmp = left_vec.xy < zero || ( left_vec.xy > right_vec.xy && right_vec.xy >= zero );
                if (any(cmp))
                {
                    left_vec = select(left_vec, right_vec, (int4) (cmp.x, cmp.y, cmp.x, cmp.y));
                    vstore4( left_vec, 0, best_buf + local_mem_offset + local_id * 4 );
                }

                left_vec = vload4(0, worst_buf + local_mem_offset + local_id * 4);
                right_vec = vload4(0, worst_buf + local_mem_offset + index * 4);
                cmp = left_vec.xy < zero || ( left_vec.xy <= right_vec.xy && right_vec.xy >= zero);
                //negative fitness values indicate that this particle has already been selected (therefore it is
                //worse than any other non-selected option left)
                if (any(cmp))
                {
                    left_vec = select(left_vec, right_vec, (int4) (cmp.x, cmp.y, cmp.x, cmp.y));
                    vstore4( left_vec, 0, worst_buf + local_mem_offset + local_id * 4 );
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (!local_id && group_id < num_swarms) //work item 0
        {
            float4 best2 = vload4(0, best_buf + local_mem_offset);
            float4 worst2 = left_vec;
            float2 best = best2.x < best2.y ? best2.xz : best2.yw;
            float2 worst = worst2.x > worst2.y ? worst2.xz : worst2.yw;

            //write the results of the reduction to the output buffers in global memory
            best_indices[group_id * num_exchange + i] = ((uint) best.y) % num_sparticles;
            worst_indices[group_id * num_exchange + i] = ((uint) worst.y) % num_sparticles;

            //replace winners' values with -1 in original buffer
            uint local_chunk_index = ((((uint) best.y) % num_sparticles) / 2) * 4;
            float4 orig = vload4(0, orig_buf + local_mem_offset + local_chunk_index);

            if ((uint) best.y == (uint) orig.z)
            {
                orig.x = -1.0f;
            }
            else
            {
                orig.y = -1.0f;
            }
            vstore2(orig.xy, 0, orig_buf + local_mem_offset + local_chunk_index);

            local_chunk_index = ((((uint) worst.y) % num_sparticles) / 2) * 4;
            orig = vload4(0, orig_buf + local_mem_offset + local_chunk_index);
            if (worst.y == orig.z)
            {
                orig.x = -1.0f;
            }
            else
            {
                orig.y = -1.0f;
            }
            vstore2(orig.xy, 0, orig_buf + local_mem_offset + local_chunk_index);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    //---------------------
    /* uint local_id = get_local_id(0); //corresponds to offset within swarm */
    /* uint group_id = get_group_id(0); //each id corresponds to one swarm (there are run->num_swarms workgroups in total) */
    /* //assign some frequently used struct params to local registers so we don't have to keep accessing (uncached) global memory */
    /* uint num_sparticles = num_sparticles; */
    /* uint fitness_offset = group_id * num_sparticles + local_id * 2; */
    /* uint i; */
    /* uint j; */
    /* uint index; */

    /* const float2 zero = (float2) (0.0f, 0.0f); */
    
    /* //copy the fitnesses of all particles in the swarm from global memory to local buffers */
    /* //note: we track fitness and fitness index using a handy two-tuple */
    /* float2 fitness_vec = vload2(0, fitness + fitness_offset); */
    /* float4 info_vec = (float4) (fitness_vec.x, fitness_vec.y, fitness_offset, fitness_offset + 1); */
    /* vstore4(info_vec, 0, orig_buf + local_id * 4); */

    /* vstore2(info_vec.zw, 0, best_buf + local_id * 4 + 2); */
    /* vstore2(info_vec.zw, 0, worst_buf + local_id * 4 + 2); */

    /* float4 left_vec; */
    /* float4 right_vec; */
    /* int2 cmp; */
    
    /* //perform num_exchange parallel reductions, each time finding one worst and one best index */
    /* for (i = 0; i < num_exchange; i++) */
    /* { */
    /*     //bring in -1.0s from previous iteration (or init worst and best bufs if 1st iteration) */
    /*     vstore4(vload4(0, orig_buf + local_id * 4), 0, worst_buf + local_id * 4); */
    /*     vstore4(vload4(0, orig_buf + local_id * 4), 0, best_buf + local_id * 4); */
        
    /*     barrier(CLK_LOCAL_MEM_FENCE); */

    /*     for (j = num_sparticles >> 2; j > 0; j >>= 1) */
    /*     { */
    /*         if (local_id < j) */
    /*         { */
    /*             index = local_id + j; */

    /*             //negative fitness values indicate that this particle has already been selected (therefore it is */
    /*             //better than any other non-selected option left) */
    /*             left_vec = vload4(0, best_buf + local_id * 4); //only load the first 2 elements (the fitnesses) */
    /*             right_vec = vload4(0, best_buf + index * 4); */
    /*             cmp = left_vec.xy < zero || ( left_vec.xy > right_vec.xy && right_vec.xy >= zero ); */
    /*             if (any(cmp)) */
    /*             { */
    /*                 left_vec = select(left_vec, right_vec, (int4) (cmp.x, cmp.y, cmp.x, cmp.y)); */
    /*                 vstore4( left_vec, 0, best_buf + local_id * 4 ); */
    /*             } */

    /*             left_vec = vload4(0, worst_buf + local_id * 4); */
    /*             right_vec = vload4(0, worst_buf + index * 4); */
    /*             cmp = left_vec.xy < zero || ( left_vec.xy <= right_vec.xy && right_vec.xy >= zero); */
    /*             //negative fitness values indicate that this particle has already been selected (therefore it is */
    /*             //worse than any other non-selected option left) */
    /*             if (any(cmp)) */
    /*             { */
    /*                 left_vec = select(left_vec, right_vec, (int4) (cmp.x, cmp.y, cmp.x, cmp.y)); */
    /*                 vstore4( left_vec, 0, worst_buf + local_id * 4 ); */
    /*             } */
    /*         } */

    /*         barrier(CLK_LOCAL_MEM_FENCE); */
    /*     } */

    /*     if (!local_id) //work item 0 */
    /*     { */
    /*         float4 best2 = vload4(0, best_buf); */
    /*         float4 worst2 = left_vec; */
    /*         float2 best = best2.x < best2.y ? best2.xz : best2.yw; */
    /*         float2 worst = worst2.x > worst2.y ? worst2.xz : worst2.yw; */

    /*         //write the results of the reduction to the output buffers in global memory */
    /*         best_indices[group_id * num_exchange + i] = ((uint) best.y) % num_sparticles; */
    /*         worst_indices[group_id * num_exchange + i] = ((uint) worst.y) % num_sparticles; */

    /*         //replace winners' values with -1 in original buffer */
    /*         uint local_chunk_index = ((((uint) best.y) % num_sparticles) / 2) * 4; */
    /*         float4 orig = vload4(0, orig_buf + local_chunk_index); */

    /*         if ((uint) best.y == (uint) orig.z) */
    /*         { */
    /*             orig.x = -1.0f; */
    /*         } */
    /*         else */
    /*         { */
    /*             orig.y = -1.0f; */
    /*         } */
    /*         vstore2(orig.xy, 0, orig_buf + local_chunk_index); */

    /*         local_chunk_index = ((((uint) worst.y) % num_sparticles) / 2) * 4; */
    /*         orig = vload4(0, orig_buf + local_chunk_index); */
    /*         if (worst.y == orig.z) */
    /*         { */
    /*             orig.x = -1.0f; */
    /*         } */
    /*         else */
    /*         { */
    /*             orig.y = -1.0f; */
    /*         } */
    /*         vstore2(orig.xy, 0, orig_buf + local_chunk_index); */
    /*     } */

    /*     barrier(CLK_LOCAL_MEM_FENCE); */
    /* } */
}
