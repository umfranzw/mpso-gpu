//this code taken from AMD sample MatrixTranspose
void __kernel init_rot_matrix_vec(
    __global float4 *output,
    __global float4 *input,
    __local  float4 *block,
    const uint width,
    const uint height,
    const uint block_size
    )
{
    /* uint globalIdx = get_global_id(0); */
/*     uint globalIdy = get_global_id(1); */
/*     uint localIdx = get_local_id(0); */
/*     uint localIdy = get_local_id(1); */

/* /\* copy from input to local memory *\/ */
/*     block[localIdy*blockSize + localIdx] =  */
/*         input[globalIdy*width + globalIdx]; */

/* /\* wait until the whole block is filled *\/ */
/*     barrier(CLK_LOCAL_MEM_FENCE); */
/*     uint groupIdx = get_group_id(0); */
/*     uint groupIdy = get_group_id(1); */

/* /\* calculate the corresponding target location for transpose */
/*    by inverting x and y values*\/ */
/*     uint targetGlobalIdx = groupIdy*blockSize + localIdy; */
/*     uint targetGlobalIdy = groupIdx*blockSize + localIdx; */

/* /\* calculate the corresponding raster indices of source and target *\/ */
/*     uint targetIndex = targetGlobalIdy*height + targetGlobalIdx; */
/*     uint sourceIndex = localIdy * blockSize + localIdx; */

/* /\* read final data from the local memory *\/ */
/*     output[targetIndex] = block[sourceIndex]; */

	uint wiWidth  = get_global_size(0);

	uint gix_t = get_group_id(0);
	uint giy_t = get_group_id(1);	

	uint num_of_blocks_x = get_num_groups(0);

	// break memory banks dependency by "reshuffling" global indeces
	uint giy = gix_t;
	uint gix = (gix_t+giy_t)%num_of_blocks_x;

	uint lix = get_local_id(0);
	uint liy = get_local_id(1);

	uint blockSize = get_local_size(0);

	uint ix = gix*blockSize + lix;
	uint iy = giy*blockSize + liy;
	uint index_in = ix + (iy)*wiWidth*4;

	// coalesced copy from input global memory into LDS
	uint ind = liy*blockSize*4+lix;
	block[ind]		= input[index_in];
	block[ind+blockSize]	= input[index_in+wiWidth];
	block[ind+blockSize*2] = input[index_in+wiWidth*2];
	block[ind+blockSize*3] = input[index_in+wiWidth*3];
		
	// wait until the whole block is filled
	barrier(CLK_LOCAL_MEM_FENCE);
	
    // calculate the corresponding target 
	// as location inside block of transposed location
	ix = giy*blockSize + lix;
	iy = gix*blockSize + liy;
	uint index_out = ix + (iy)*wiWidth*4;

	ind = lix*blockSize*4+liy;
	float4 v0 = block[ind];
	float4 v1 = block[ind+blockSize];
	float4 v2 = block[ind+blockSize*2];
	float4 v3 = block[ind+blockSize*3];
	
	// coalesced copy of transposed data in LDS into output global memory
	output[index_out]			= (float4)(v0.x, v1.x, v2.x, v3.x);
	output[index_out+wiWidth]	= (float4)(v0.y, v1.y, v2.y, v3.y);
	output[index_out+wiWidth*2]	= (float4)(v0.z, v1.z, v2.z, v3.z);
	output[index_out+wiWidth*3]	= (float4)(v0.w, v1.w, v2.w, v3.w);
}
