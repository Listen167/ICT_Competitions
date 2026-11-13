#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import torch
from ml_dtypes import bfloat16

def impl(input_values, concat_dim):
    """
    ConcatD 算子的预期函数实现
    沿指定维度拼接多个张量
    
    Args:
        input_values: 输入张量列表，每个张量可以是 numpy 数组或 torch tensor
        concat_dim: 拼接维度（int 类型）
    
    Returns:
        拼接后的 numpy 数组
    """
    # ------------------------------
    # 处理 concat_dim 参数
    # ------------------------------
    if isinstance(concat_dim, np.ndarray):
        concat_dim = concat_dim.item()
    
    # 确保 concat_dim 是整数
    concat_dim = int(concat_dim)
    
    # ------------------------------
    # 转换输入为 torch tensor
    # ------------------------------
    torch_inputs = []
    original_dtypes = []
    
    for i, x in enumerate(input_values):
        # 保存原始数据类型
        if isinstance(x, np.ndarray):
            original_dtype = x.dtype
            original_dtypes.append(original_dtype)
            
            # 处理 bfloat16 的特殊情况
            if x.dtype == bfloat16:
                # 绕开 pytorch 不支持直接导入 ml_dtypes.bfloat16 的问题
                x = torch.tensor(x.astype(np.float32)).to(torch.bfloat16)
            else:
                x = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            original_dtype = x.dtype
            original_dtypes.append(original_dtype)
            x = x.detach().cpu()
        else:
            raise TypeError(f"Input {i} must be numpy array or torch tensor, got {type(x)}")
        
        torch_inputs.append(x)
    
    # ------------------------------
    # 检查所有输入的数据类型是否一致
    # ------------------------------
    first_dtype = original_dtypes[0]
    for i, dtype in enumerate(original_dtypes):
        if dtype != first_dtype:
            raise ValueError(f"All input tensors must have the same dtype. "
                           f"Input 0 has dtype {first_dtype}, but input {i} has dtype {dtype}")
    
    # ------------------------------
    # 沿指定维度拼接张量
    # ------------------------------
    # 处理负索引
    if concat_dim < 0:
        num_dims = torch_inputs[0].dim()
        concat_dim = num_dims + concat_dim
    
    # 验证 concat_dim 的有效性
    num_dims = torch_inputs[0].dim()
    if concat_dim < 0 or concat_dim >= num_dims:
        raise ValueError(f"concat_dim {concat_dim} is out of range for tensor with {num_dims} dimensions")
    
    # 验证所有输入的维度数相同
    for i, x in enumerate(torch_inputs):
        if x.dim() != num_dims:
            raise ValueError(f"All input tensors must have the same number of dimensions. "
                           f"Input 0 has {num_dims} dimensions, but input {i} has {x.dim()} dimensions")
    
    # 验证除拼接维度外的其他维度是否匹配
    for i, x in enumerate(torch_inputs):
        for dim in range(num_dims):
            if dim != concat_dim:
                if x.shape[dim] != torch_inputs[0].shape[dim]:
                    raise ValueError(f"Input tensors must have the same shape in all dimensions "
                                   f"except the concatenation dimension. "
                                   f"Dimension {dim}: input 0 has size {torch_inputs[0].shape[dim]}, "
                                   f"but input {i} has size {x.shape[dim]}")
    
    # 执行拼接
    result = torch.cat(torch_inputs, dim=concat_dim)
    
    # ------------------------------
    # 转换为 numpy 数组输出
    # ------------------------------
    if result.dtype is torch.bfloat16:
        return result.to(torch.float32).numpy().astype(bfloat16)
    else:
        return result.numpy()


if __name__ == "__main__":
    # 测试代码
    print("=== ConcatD 预期函数测试 ===\n")
    
    # Case 1: 1D 张量拼接
    print("Case 1: 1D 张量拼接")
    x1_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x1_2 = np.array([4.0, 5.0], dtype=np.float32)
    y1 = impl([x1_1, x1_2], 0)
    print(f"输入1: {x1_1}, 输入2: {x1_2}, concat_dim=0")
    print(f"输出: {y1}, 形状: {y1.shape}, 数据类型: {y1.dtype}\n")
    
    # Case 2: 2D 张量沿第0轴拼接
    print("Case 2: 2D 张量沿第0轴拼接")
    x2_1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2_2 = np.array([[5.0, 6.0]], dtype=np.float32)
    y2 = impl([x2_1, x2_2], 0)
    print(f"输入1:\n{x2_1}\n输入2:\n{x2_2}, concat_dim=0")
    print(f"输出:\n{y2}, 形状: {y2.shape}, 数据类型: {y2.dtype}\n")
    
    # Case 3: 2D 张量沿第1轴拼接
    print("Case 3: 2D 张量沿第1轴拼接")
    x3_1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x3_2 = np.array([[5.0], [6.0]], dtype=np.float32)
    y3 = impl([x3_1, x3_2], 1)
    print(f"输入1:\n{x3_1}\n输入2:\n{x3_2}, concat_dim=1")
    print(f"输出:\n{y3}, 形状: {y3.shape}, 数据类型: {y3.dtype}\n")
    
    # Case 4: float16 数据类型
    print("Case 4: float16 数据类型")
    x4_1 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    x4_2 = np.array([4.0, 5.0], dtype=np.float16)
    y4 = impl([x4_1, x4_2], 0)
    print(f"输入1: {x4_1}, 输入2: {x4_2}, concat_dim=0")
    print(f"输出: {y4}, 形状: {y4.shape}, 数据类型: {y4.dtype}\n")
    
    # Case 5: bfloat16 数据类型
    print("Case 5: bfloat16 数据类型")
    x5_1 = np.array([1.0, 2.0, 3.0], dtype=bfloat16)
    x5_2 = np.array([4.0, 5.0], dtype=bfloat16)
    y5 = impl([x5_1, x5_2], 0)
    print(f"输入1: {x5_1}, 输入2: {x5_2}, concat_dim=0")
    print(f"输出: {y5}, 形状: {y5.shape}, 数据类型: {y5.dtype}\n")
    
    # Case 6: 多个输入拼接
    print("Case 6: 多个输入拼接")
    x6_1 = np.array([1.0, 2.0], dtype=np.float32)
    x6_2 = np.array([3.0, 4.0], dtype=np.float32)
    x6_3 = np.array([5.0, 6.0], dtype=np.float32)
    y6 = impl([x6_1, x6_2, x6_3], 0)
    print(f"输入1: {x6_1}, 输入2: {x6_2}, 输入3: {x6_3}, concat_dim=0")
    print(f"输出: {y6}, 形状: {y6.shape}, 数据类型: {y6.dtype}\n")
    
    # Case 7: 3D 张量拼接
    print("Case 7: 3D 张量拼接")
    x7_1 = np.random.uniform(-5, 5, [2, 3, 4]).astype(np.float32)
    x7_2 = np.random.uniform(-5, 5, [2, 3, 6]).astype(np.float32)
    y7 = impl([x7_1, x7_2], 2)
    print(f"输入1形状: {x7_1.shape}, 输入2形状: {x7_2.shape}, concat_dim=2")
    print(f"输出形状: {y7.shape}, 数据类型: {y7.dtype}\n")
    
    # Case 8: 4D 张量拼接
    print("Case 8: 4D 张量拼接")
    x8_1 = np.random.uniform(-5, 5, [2, 3, 4, 5]).astype(np.float32)
    x8_2 = np.random.uniform(-5, 5, [2, 3, 4, 7]).astype(np.float32)
    y8 = impl([x8_1, x8_2], 3)
    print(f"输入1形状: {x8_1.shape}, 输入2形状: {x8_2.shape}, concat_dim=3")
    print(f"输出形状: {y8.shape}, 数据类型: {y8.dtype}\n")
    
    # Case 9: 负索引 concat_dim
    print("Case 9: 负索引 concat_dim")
    x9_1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x9_2 = np.array([[5.0], [6.0]], dtype=np.float32)
    y9 = impl([x9_1, x9_2], -1)
    print(f"输入1:\n{x9_1}\n输入2:\n{x9_2}, concat_dim=-1")
    print(f"输出:\n{y9}, 形状: {y9.shape}, 数据类型: {y9.dtype}\n")
    
    # Case 10: int32 数据类型
    print("Case 10: int32 数据类型")
    x10_1 = np.array([1, 2, 3], dtype=np.int32)
    x10_2 = np.array([4, 5], dtype=np.int32)
    y10 = impl([x10_1, x10_2], 0)
    print(f"输入1: {x10_1}, 输入2: {x10_2}, concat_dim=0")
    print(f"输出: {y10}, 形状: {y10.shape}, 数据类型: {y10.dtype}\n")
    
    print("=== 所有测试用例通过！===")