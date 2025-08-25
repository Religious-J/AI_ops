from safetensors import safe_open
import numpy as np
import torch
import os

# 输入 safetensors 文件路径
safetensors_file_path = "model-00000-of-00014.safetensors"

# 输出目录（按层保存）
output_dir = "output_npy"
os.makedirs(output_dir, exist_ok=True)

# 只提取名字里包含这个关键字的张量
filter_keyword = "model.layers.26.mlp"

try:
    print(f"正在从 '{safetensors_file_path}' 中提取张量 (过滤关键字: '{filter_keyword}')...")
    with safe_open(safetensors_file_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        matched_keys = [k for k in keys if filter_keyword in k]

        if not matched_keys:
            print(f"⚠️ 没有找到包含关键字 '{filter_keyword}' 的张量。")
        else:
            for tensor_name in matched_keys:
                print(f"处理张量: {tensor_name}")
                tensor = f.get_tensor(tensor_name)

                # 如果是 bfloat16 转换到 float16
                if tensor.dtype == torch.bfloat16:
                    print(f"  - 检测到 bfloat16，转换为 float16")
                    tensor = tensor.to(torch.float16)

                # 转 numpy
                arr = tensor.cpu().numpy()

                # 构造输出文件路径（把 "." 换成 "_"，避免文件名太乱）
                safe_name = tensor_name.replace(".", "_")
                output_path = os.path.join(output_dir, f"{safe_name}.npy")

                np.save(output_path, arr)
                print(f"  - 已保存到 {output_path} (形状: {arr.shape}, 类型: {arr.dtype})")

except FileNotFoundError:
    print(f"❌ 错误: 文件 '{safetensors_file_path}' 不存在，请检查路径。")
except Exception as e:
    print(f"❌ 处理文件时发生错误: {e}")
