#include "sample_support.h"
#include "aclnn_concat_d.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor input_0_0({3}, ACL_FLOAT, 12, "input_values_0.bin", true, false, root);
            Tensor input_0_1({5}, ACL_FLOAT, 20, "input_values_1.bin", true, false, root);
            TensorList input_0_list({input_0_0.tensor,input_0_1.tensor});
            Tensor output_0_0({8}, ACL_FLOAT, 32, "output_data.bin", false, false, root);
            int64_t attr_0 = 0LL;
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnConcatDGetWorkspaceSize(
                input_0_list.list,
                attr_0,
                output_0_0.tensor,
                &size,
                &executor));
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            check(aclnnConcatD(workspace, size, executor, stream));
            check(aclrtSynchronizeStream(stream));
            output_0_0.save(root);
            if (workspace)
                check(aclrtFree(workspace));
        }
        check(aclrtDestroyStream(stream));
        check(aclrtResetDevice(device));
        check(aclFinalize());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
}
