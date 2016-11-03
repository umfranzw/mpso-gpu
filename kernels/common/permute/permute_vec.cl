//Note - bitonic sort code below is adapted (vectorized) from an AMD sample,
//which requires the inclusion of this disclaimer.
/* ============================================================

Copyright (c) 2011 Advanced Micro Devices, Inc.  All rights reserved.

Redistribution and use of this material is permitted under the following 
conditions:

Redistributions must retain the above copyright notice and all terms of this 
license.

In no event shall anyone redistributing or accessing or using this material 
commence or participate in any arbitration or legal action relating to this 
material against Advanced Micro Devices, Inc. or any copyright holders or 
contributors. The foregoing shall survive any expiration or termination of 
this license or any agreement or access or use related to this material. 

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION 
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT 
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY 
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO 
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERA TION, OR THAT IT IS FREE 
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER 
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED 
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, 
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT. 
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR 
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY 
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES, 
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS 
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS 
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND 
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES, 
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE 
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE 
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR 
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR 
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE 
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL 
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR 
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS 
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO 
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER 
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH 
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS 
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S. 
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS, 
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS, 
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS. 
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY 
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is 
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to 
computer software and technical data, respectively. Use, duplication, 
distribution or disclosure by the U.S. Government and/or DOD agencies is 
subject to the full extent of restrictions in all applicable regulations, 
including those found at FAR52.227 and DFARS252.227 et seq. and any successor 
regulations thereof. Use of this material by the U.S. Government and/or DOD 
agencies is acknowledgment of the proprietary rights of any copyright holders 
and contributors, including those of Advanced Micro Devices, Inc., as well as 
the provisions of FAR52.227-14 through 23 regarding privately developed and/or 
commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and 
supersedes all proposals and prior discussions and writings between the parties 
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be 
modified or waived, and no breach of this license can be excused, unless done 
so in a writing signed by all affected parties. Each term of this license is 
separately enforceable. If any term of this license is determined to be or 
becomes unenforceable or illegal, such term shall be reformed to the minimum 
extent necessary in order for this license to remain in effect in accordance 
with its terms as modified by such reformation. This license shall be governed 
by and construed in accordance with the laws of the State of Texas without 
regard to rules on conflicts of law of any state or jurisdiction or the United 
Nations Convention on the International Sale of Goods. All disputes arising out 
of this license shall be subject to the jurisdiction of the federal and state 
courts in Austin, Texas, and all defenses are hereby waived concerning personal 
jurisdiction and venue of these courts.

============================================================ */


//s workgorups of p / 4 threads each

uint2 threefish_encrypt(
    const uint2 key,
    const uint2 val
    )
{
    //we only use the first 2 rands out of the counter c (Random123 doesn't have a 2x32 type).
    threefry4x32_key_t k = {key.x, key.y, 0, 0};
    threefry4x32_ctr_t c = {{}};
    threefry4x32_ctr_t result;

    c.v[0] = val.x;
    c.v[1] = val.y;
    result = threefry4x32(c, k);

    return (uint2) (result.v[0], result.v[1]);
}

void init_memory(
    const uint local_id,
    __local uint *scratch_vals,
    __local uint *scratch_indices,
    const uint launch_num,
    const uint seed,
    const uint num_sparticles
    )
{
    const uint local_size = get_local_size(0);
    const uint group_id = get_group_id(0);
    //there should be a gap between key values on each kernel launch
    //there should also be a gap between the key values of different workgroups
    const uint2 key = (uint2) (
        seed + launch_num * get_num_groups(0) * 2 + group_id,
        seed + launch_num * get_num_groups(0) * 2 + group_id + 1
        );

    //generate and store encrypted values
    uint2 vals = (uint2) (
        local_id * 2,
        local_id * 2 + 1
        );

    //vals = threefish_encrypt(key, vals);

    vstore2(
        vals,
        0,
        scratch_vals + local_id * 2
        );

    //store indices
    vstore2(
        (uint2) (local_id * 2, local_id * 2 + 1),
        0,
        scratch_indices + local_id * 2
        );
    
    barrier(CLK_LOCAL_MEM_FENCE);
}

void bitonic_sort(
    const uint local_id,
    const uint len,
    __local uint *scratch_vals,
    __local uint *scratch_indices
    )
{
    //bitonic sort code, adapted (vectorized) from AMD sample
    uint num_stages = 0;
    uint i;
    //log(len) stages
    for (i = 1; i < len; i *= 2)
    {
        num_stages++;
    }

    uint s;
    uint p;
    uint sort_dir = 1; //1 = asc, 0 = desc
    uint pass_sort_dir;
    uint pair_dist;
    uint block_width;
    uint same_dir_block_width;
    uint lesser_val;
    uint greater_val;
    uint lesser_index;
    uint greater_index;
    uint left_id;
    uint right_id;
    uint left_val;
    uint right_val;
    uint left_index;
    uint right_index;

    for (s = 0; s < num_stages; s++)
    {
        for (p = 0; p < s + 1; p++)
        {
            /* thread_ids = (uint2) (local_id * 2, local_id * 2 + 1); */
            
            /* pass_sort_dir = (uint2) (sort_dir, sort_dir); */
            /* pair_dist = 1 << (s - p); */
            /* block_width = pair_dist * 2; */

            /* ids.xy = (thread_ids % pair_dist) + (thread_ids / pair_dist) * block_width; */
            /* ids.zw = ids.xy + pair_dist; */
            
            /* same_dir_block_width = 1 << s; */

            /* int2 cmp = (convert_int2(thread_ids) / (int2) same_dir_block_width) % 2; */
            /* pass_sort_dir = convert_uint2( select(convert_int2(pass_sort_dir), !convert_int2(pass_sort_dir), cmp) ); */

            /* vals.xy = vload2(0, scratch_vals + ids.x); */
            /* indices.xy = vload2(0, scratch_indices + ids.x); */
            /* int index_swap_needed = (s != p); */
            /* uint right_vec_id = index_swap_needed ? ids.y : ids.z; */
            /* vals.zw = vload2(0, scratch_vals + right_vec_id); */
            /* indices.zw = vload2(0, scratch_vals + right_vec_id); */

            /* vals = index_swap_needed ? vals.xzyw : vals; */
            /* indices = index_swap_needed ? indices.xzwy : indices; */
            
            /* left_swap = (pass_sort_dir.x && vals.x > vals.y) || (!pass_sort_dir.x && left_vals.x < left_vals.y); */
            /* right_swap = (pass_sort_dir.y && vals.z > vals.w) || (!pass_sort_dir.y && vals.z < vals.w); */

            /* //vals.xy = select(vals.xy, vals.yx, left_swap); */
            /* vals.xy = left_swap ? vals.yx : vals.xy; */
            /* //vals.zw = select(vals.zw, vals.wz, right_swap); */
            /* vals.zw = right_swap ? vals.wz : vals.zw; */
            /* //indices.xy = select(indices.xy, indices.yx, left_swap); */
            /* indices.xy = left_swap ? indices.yx : indices.xy; */
            /* //indices.zw = select(indices.zw, indices.wz, right_swap); */
            /* indices.zw = right_swap ? indices.wz : indices.zw; */

            /* vals = index_swap_needed ? vals.xzwy : vals; */
            /* indices = index_swap_needed ? indices.xzwy : indices; */

            /* vstore2(vals.xy, 0, scratch_vals + ids.x); */
            /* vstore2(vals.zw, 0, scratch_vals + right_vec_id); */
            /* vstore2(indices.xy, 0, scratch_indices + ids.x); */
            /* vstore2(indices.zw, 0, scratch_indices + right_vec_id); */

            /* barrier(CLK_LOCAL_MEM_FENCE); */
            
            pass_sort_dir = sort_dir;
            pair_dist = 1 << (s - p);
            block_width = pair_dist * 2;
            left_id = ( (local_id % pair_dist) + (local_id / pair_dist) * block_width );
            right_id = left_id + pair_dist;
            same_dir_block_width = 1 << s;

            if ( (local_id / same_dir_block_width) % 2 )
            {
                pass_sort_dir = !pass_sort_dir;
            }

            left_val = scratch_vals[left_id];
            right_val = scratch_vals[right_id];
            left_index = scratch_indices[left_id];
            right_index = scratch_indices[right_id];

            lesser_val = left_val < right_val ? left_val : right_val;
            lesser_index = left_val < right_val ? left_index : right_index;
            greater_val = left_val < right_val ? right_val : left_val;
            greater_index = left_val < right_val ? right_index : left_index;

            scratch_vals[left_id] = pass_sort_dir ? lesser_val : greater_val;
            scratch_indices[left_id] = pass_sort_dir ? lesser_index : greater_index;
            scratch_vals[right_id] = pass_sort_dir ? greater_val : lesser_val;
            scratch_indices[right_id] = pass_sort_dir ? greater_index : lesser_index;

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

void __kernel permute_vec(
    __global uint *list,
    __local uint *scratch_vals,
    __local uint *scratch_indices,
    const uint launch_num,
    const uint seed,
    const uint num_sparticles
    )
{
    const uint local_id = get_local_id(0);
    const uint len = get_local_size(0) * 2;

    init_memory(
        local_id,
        scratch_vals,
        scratch_indices,
        launch_num,
        seed,
        num_sparticles
        );
    
    bitonic_sort(
        local_id,
        len,
        scratch_vals,
        scratch_indices
        );
    
    async_work_group_copy(
        list + get_group_id(0) * len,
        scratch_indices,
        len,
        0
        );
}
