/*!
 * \file test_concat_d.cpp
 * \brief ConcatD 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "concat_d_tiling.h"
#include "../../../op_kernel/concat_d.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class ConcatDKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ConcatDKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ConcatDKernelTest TearDown" << endl;
    }
};

TEST_F(ConcatDKernelTest, test_kernel_run)
{
    constexpr size_t size = 3;
    constexpr size_t tilingDataSize = sizeof(ConcatDTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t input_values_t0ByteSize = 3 * 4;
    constexpr size_t input_values_t1ByteSize = 5 * 4;
    constexpr size_t input_valuesByteSize = 56;
    constexpr size_t output_dataByteSize = 8 * 4;
    std::vector<uint8_t> input_values_t0Host(input_values_t0ByteSize);
    std::vector<uint8_t> input_values_t1Host(input_values_t1ByteSize);
    std::vector<uint8_t> output_dataHost(output_dataByteSize);
    
    
    ASSERT_NE(std::getenv("SAMPLE_DIR"), nullptr);
    const std::string root = std::getenv("SAMPLE_DIR");
    uint8_t* input_values_t0 = (uint8_t*)AscendC::GmAlloc(input_values_t0ByteSize);
    uint8_t* input_values_t1 = (uint8_t*)AscendC::GmAlloc(input_values_t1ByteSize);
    uint8_t* input_values = (uint8_t*)AscendC::GmAlloc(input_valuesByteSize);
    uint8_t* output_data = (uint8_t*)AscendC::GmAlloc(output_dataByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    { std::ifstream f(root + "/input/input_values_0.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(input_values_t0), 12); ASSERT_EQ(f.gcount(), 12); }
    { std::ifstream f(root + "/input/input_values_1.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(input_values_t1), 20); ASSERT_EQ(f.gcount(), 20); }
    
    uint64_t input_valuesDesc[] = {40, 1, 3, 4294967297, 5, reinterpret_cast<uint64_t>(input_values_t0), reinterpret_cast<uint64_t>(input_values_t1)};
    memcpy(input_values, input_valuesDesc, sizeof(input_valuesDesc));
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/concat_d_tiling_data.h 中的字段定义
    //         参考 op_host/concat_d_tiling.cpp 中的 tiling 计算逻辑
    ConcatDTilingData* tilingData = reinterpret_cast<ConcatDTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/concat_d_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((concat_d<1>), numBlocks, input_values, output_data, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(output_dataHost.data(), output_data, output_dataByteSize);
    { std::ofstream _ofs((root + "/output/output_data.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(output_dataHost.data()), output_dataByteSize); }
    
    AscendC::GmFree(input_values_t0);
    AscendC::GmFree(input_values_t1);
    AscendC::GmFree(input_values);
    AscendC::GmFree(output_data);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
