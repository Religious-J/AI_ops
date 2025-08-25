import torch
import numpy as np

# 模拟 FP4_VALUES 查找表
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

def simulate_mxfp4_tensor_copy():
    print("=== _get_mxfp4_tensor_copy 输入输出模拟 ===\n")

    # ========== 输入数据 ==========
    print("1. 输入数据:")

    # 模拟 blocks 数据 (每个字节包含2个4-bit值)
    # 假设是一个小的权重矩阵: [2层, 3行, 2列] (实际存储时列数减半)
    loaded_blocks = torch.tensor([
        [[0xB3, 0x47],   # 第1层，第1行: B3=11|3, 47=4|7
         [0x82, 0xC1],   # 第1层，第2行: 82=8|2, C1=12|1
         [0x95, 0x6A]],  # 第1层，第3行: 95=9|5, 6A=6|10

        [[0x71, 0x3E],   # 第2层，第1行: 71=7|1, 3E=3|14
         [0x40, 0x5D],   # 第2层，第2行: 40=4|0, 5D=5|13
         [0x8F, 0x26]]   # 第2层，第3行: 8F=8|15, 26=2|6
    ], dtype=torch.uint8)

    print(f"loaded_blocks.shape: {loaded_blocks.shape}")
    print(f"loaded_blocks (hex):\n{np.array([[f'{x:02X}' for x in row] for layer in loaded_blocks for row in layer]).reshape(2, 3, 2)}")

    # 模拟 scales 数据 (每个块的共享指数)
    # 注意: scales 的形状应该和 blocks 的前几维一致，但没有最后一维
    # blocks: [2, 3, 2] → scales: [2, 3] (每行一个scale)
    loaded_scales = torch.tensor([
        [130, 125, 129],  # 第1层3行: 指数为 3, -2, 2
        [127, 132, 124]   # 第2层3行: 指数为 0, 5, -3
    ], dtype=torch.uint8)

    print(f"\nloaded_scales (原始): {loaded_scales}")
    print(f"loaded_scales (减去127后): {loaded_scales.int() - 127}")

    # ========== 第1步: 分离高低位 ==========
    print("\n2. 第1步 - 分离高低位:")

    loaded_blocks_lo = loaded_blocks & 0x0F  # 提取低4位
    loaded_blocks_hi = loaded_blocks >> 4    # 提取高4位

    print(f"低4位: {loaded_blocks_lo}")
    print(f"高4位: {loaded_blocks_hi}")

    # 详细展示第一个字节的分离过程
    # first_byte = 0xB3
    # print(f"\n详细展示: 0x{first_byte:02X} = {first_byte:08b}")
    # print(f"  低4位: {first_byte & 0x0F:04b} = {first_byte & 0x0F}")
    # print(f"  高4位: {first_byte >> 4:04b} = {first_byte >> 4}")

    # ========== 第2步: 交错排列 ==========
    print("\n3. 第2步 - 交错排列 (为SwiGLU优化):")

    loaded_blocks_stacked = torch.stack((loaded_blocks_lo, loaded_blocks_hi), dim=-1)
    print(f"stack后形状: {loaded_blocks_stacked.shape}")
    print(f"stack后数据:\n{loaded_blocks_stacked}")

    loaded_blocks_interleaved = loaded_blocks_stacked.view(*loaded_blocks_stacked.shape[:-2], -1)
    print(f"交错排列后形状: {loaded_blocks_interleaved.shape}")
    print(f"交错排列后数据:\n{loaded_blocks_interleaved}")

    # 展示交错模式
    # print("\n交错模式展示 (以第1层第1行为例):")
    # row = loaded_blocks_interleaved[0, 0]  # [3, 11, 7, 4]
    # print(f"原始: [低4位, 高4位, 低4位, 高4位] = {row.tolist()}")
    # print("这样排列使得SwiGLU可以分离为:")
    # print(f"  W权重 (偶数位): {row[0::2].tolist()}")  # [3, 7]
    # print(f"  V权重 (奇数位): {row[1::2].tolist()}")  # [11, 4]

    # ========== 第3步: 处理缩放因子 ==========
    print("\n4. 第3步 - 处理缩放因子:")

    processed_scales = loaded_scales.int() - 127
    print(f"处理后的scales: {processed_scales}")

    # ========== 第4步: 查表解码 ==========
    print("\n5. 第4步 - 查表解码:")

    print("FP4 查找表:")
    for i, val in enumerate(FP4_VALUES):
        print(f"  索引 {i:2d}: {val:+4.1f}")

    # 创建查找表
    fp4_values = torch.tensor(FP4_VALUES, dtype=torch.bfloat16)

    # 查找基础值
    base_values = fp4_values[loaded_blocks_interleaved.long()]
    print(f"\n查表后的基础值:\n{base_values}")

    # ========== 第5步: 应用缩放 ==========
    print("\n6. 第5步 - 应用缩放:")

    # 为广播添加维度
    scales_for_broadcast = processed_scales.unsqueeze(-1)  # [2, 3, 1]
    print(f"广播用的scales形状: {scales_for_broadcast.shape}")
    print(f"广播用的scales:\n{scales_for_broadcast}")

    # 应用 ldexp (value * 2^exponent)
    final_values = torch.ldexp(base_values, scales_for_broadcast)
    print(f"\n最终解码值:\n{final_values}")

    # ========== 详细计算示例 ==========
    print("\n7. 详细计算示例 (第1层第1行):")

    layer0_row0_indices = loaded_blocks_interleaved[0, 0]  # [3, 11, 7, 4]
    layer0_row0_scale = processed_scales[0, 0]  # 3

    print(f"4-bit索引: {layer0_row0_indices.tolist()}")
    print(f"该行的共享指数: {layer0_row0_scale}")
    print("计算过程:")

    for i, idx in enumerate(layer0_row0_indices):
        base_val = FP4_VALUES[idx]
        final_val = base_val * (2 ** layer0_row0_scale)
        print(f"  位置{i}: FP4_VALUES[{idx}] * 2^{layer0_row0_scale} = {base_val} * {2**layer0_row0_scale} = {final_val}")

    # ========== 最终形状整理 ==========
    print("\n8. 最终形状整理:")

    final_tensor = final_values.view(*final_values.shape[:-2], -1)
    print(f"最终输出形状: {final_tensor.shape}")
    print(f"最终输出值:\n{final_tensor}")

    # ========== 总结 ==========
    print("\n9. 总结:")
    print(f"输入: blocks{loaded_blocks.shape} + scales{loaded_scales.shape}")
    print(f"输出: {final_tensor.shape} 的浮点张量")
    print("核心变换: 4-bit索引 → FP4值 → 缩放 → 最终浮点数")
    print("交错排列使得SwiGLU可以高效分离门控权重和值权重")

    return final_tensor

# 运行模拟
if __name__ == "__main__":
    result = simulate_mxfp4_tensor_copy()
