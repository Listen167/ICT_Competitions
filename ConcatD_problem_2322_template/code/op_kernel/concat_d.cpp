/*!
 * \file concat_d.cpp
 * \brief ConcatD 算子 kernel 入口
 */

#include "concat_d.h"

enum class ConcatDTilingKey : uint32_t
{
    TILING_KEY_CONCATD_MODE_0 = 0,
    TILING_KEY_CONCATD_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void concat_d(GM_ADDR input_values, GM_ADDR output_data, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ConcatDTilingData);
    GET_TILING_DATA_WITH_STRUCT(ConcatDTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ConcatDTilingKey::TILING_KEY_CONCATD_MODE_0)) {
        NsConcatD::ConcatD<half> op;
        op.Init(input_values, output_data, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ConcatDTilingKey::TILING_KEY_CONCATD_MODE_1)) {
        NsConcatD::ConcatD<float> op;
        op.Init(input_values, output_data, &tilingData);
        op.Process();
    }
}
