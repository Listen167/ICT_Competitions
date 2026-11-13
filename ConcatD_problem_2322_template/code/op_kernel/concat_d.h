/*!
 * \file concat_d.h
 * \brief ConcatD 算子 kernel 类定义
 */

#ifndef CONCATD_H
#define CONCATD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "concat_d_tiling_data.h"
#include "concat_d_tiling_key.h"

namespace NsConcatD {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class ConcatD {
public:
    __aicore__ inline ConcatD(){};

    __aicore__ inline void Init(GM_ADDR input_values, GM_ADDR output_data, const ConcatDTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void ConcatD<T>::Init(GM_ADDR input_values, GM_ADDR output_data, const ConcatDTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
}

template <typename T>
__aicore__ inline void ConcatD<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
}

template <typename T>
__aicore__ inline void ConcatD<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
}

template <typename T>
__aicore__ inline void ConcatD<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
}

template <typename T>
__aicore__ inline void ConcatD<T>::Process()
{
    // TODO: 实现 Process 逻辑
}

} // namespace NsConcatD
#endif // CONCATD_H
