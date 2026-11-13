#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "concat_d_tiling_data.h"

namespace ConcatDUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "ConcatD";

struct ConcatDTestParam {
    std::string caseName;
    std::initializer_list<int64_t> input_values_t0Shape;
    ge::DataType input_values_t0Dtype;
    ge::Format input_values_t0Format;
    std::initializer_list<int64_t> input_values_t1Shape;
    ge::DataType input_values_t1Dtype;
    ge::Format input_values_t1Format;
    std::initializer_list<int64_t> output_dataShape;
    ge::DataType output_dataDtype;
    ge::Format output_dataFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/concat_d_tiling_key.h 和 op_host/concat_d_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/concat_d_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/concat_d_tiling.cpp 中 GetWorkspaceSize 的逻辑
static ConcatDTestParam testCases[] = {
    {"concat_d_0", {3}, ge::DT_FLOAT, ge::FORMAT_ND, {5}, ge::DT_FLOAT, ge::FORMAT_ND, {8}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class ConcatDTilingTest : public testing::TestWithParam<ConcatDTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "ConcatDTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "ConcatDTilingTest TearDown." << std::endl;
    }
};

struct ConcatDCompileInfo {} compileInfo;

static void TestOneParamCase(const ConcatDTestParam &param)
{
    gert::StorageShape input_values_t0Shape = {param.input_values_t0Shape, param.input_values_t0Shape};
    gert::StorageShape input_values_t1Shape = {param.input_values_t1Shape, param.input_values_t1Shape};
    gert::StorageShape output_dataShape = {param.output_dataShape, param.output_dataShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{input_values_t0Shape, param.input_values_t0Dtype, param.input_values_t0Format},
        {input_values_t1Shape, param.input_values_t1Dtype, param.input_values_t1Format}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{output_dataShape, param.output_dataDtype, param.output_dataFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(ConcatDTilingTest, tiling_test)
{
    const ConcatDTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    ConcatDTilingTests,
    ConcatDTilingTest,
    testing::ValuesIn(testCases));

}
