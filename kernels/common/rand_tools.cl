float4 get_float_rands_vec(
    uint global_id,
    uint stream_num,
    uint call_num,
    uint seed
    )
{
    philox4x32_key_t k = {{global_id, call_num}};
    philox4x32_ctr_t c = {{seed, stream_num}};

    philox4x32_ctr_t result = philox4x32(c, k);

    return (float4) (u01_closed_closed_32_24(result.v[0]),
                     u01_closed_closed_32_24(result.v[1]),
                     u01_closed_closed_32_24(result.v[2]),
                     u01_closed_closed_32_24(result.v[3]));
}

uint4 get_uint_rands_vec(
    uint global_id,
    uint stream_num,
    uint call_num,
    uint seed
    )
{
    philox4x32_key_t k = {{global_id, call_num}};
    philox4x32_ctr_t c = {{seed, stream_num}};

    philox4x32_ctr_t result = philox4x32(c, k);

    return (uint4) (result.v[0],
                    result.v[1],
                    result.v[2],
                    result.v[3]);
}
