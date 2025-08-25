import torch
import numpy as np

FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)

def dequant_mxfp4(blocks, scales, dtype: torch.dtype = torch.float16):

    loaded_blocks_lo = blocks & 0x0F
    loaded_blocks_hi = blocks >> 4

    loaded_blocks_stacked = torch.stack((loaded_blocks_lo, loaded_blocks_hi), dim=-1)
    loaded_blocks_interleaved = loaded_blocks_stacked.view(*loaded_blocks_stacked.shape[:-2], -1)

    # Upcast to int32 and subtract bias
    processed_scales = scales.int() - 127

    # Convert MXFP4 numbers into target dtype
    fp4_values = torch.tensor(FP4_VALUES, dtype = dtype)
    base_values = fp4_values[loaded_blocks_interleaved.long()]
    scales_for_broadcast = processed_scales.unsqueeze(-1)

    # ldexp (value * 2^exponent)
    dequant_values = torch.ldexp(base_values, scales_for_broadcast)
    dequant_tensor = dequant_values.view(*dequant_values.shape[:-2], -1)

    return dequant_tensor

def ffn_layer_mxfp4_gpu(
    gate_up_proj_blocks,
    gate_up_proj_scales,
    gate_up_proj_bias,
    down_proj_blocks,
    down_proj_scales,
    down_proj_bias,
    batch_inputs,
    weight_presion: str = "MXFP4",
    op_device: str = "cpu",  # "cpu", "gpu"
):
    batch_inputs = torch.from_numpy(batch_inputs).to(torch.float16)

    gate_up_proj_blocks = torch.from_numpy(gate_up_proj_blocks).to(torch.uint8)
    gate_up_proj_scales = torch.from_numpy(gate_up_proj_scales).to(torch.uint8)

    down_proj_blocks = torch.from_numpy(down_proj_blocks).to(torch.uint8)
    down_proj_scales = torch.from_numpy(down_proj_scales).to(torch.uint8)

    print(gate_up_proj_blocks.shape)

    mlp1_weight = dequant_mxfp4(gate_up_proj_blocks, gate_up_proj_scales)

    print(mlp1_weight.shape)

    mlp1_bias = torch.from_numpy(gate_up_proj_bias).to(torch.float16)

    mlp2_weight = dequant_mxfp4(down_proj_blocks, down_proj_scales)
    mlp2_bias = torch.from_numpy(down_proj_bias).to(torch.float16)

    mlp1_weight = mlp1_weight.transpose(0, 1)
    hidden = swiglu(torch.matmul(batch_inputs, mlp1_weight) + mlp1_bias)

    mlp2_weight = mlp2_weight.transpose(0, 1)
    out = torch.matmul(hidden, mlp2_weight) + mlp2_bias

    return out

if __name__ == "__main__":
    PATH = "/home/chen/mxfp4_data/output_npy/"

    gate_up_proj_bias = np.load(PATH + "gate_up_proj_bias.npy")
    gate_up_proj_blocks = np.load(PATH + "gate_up_proj_blocks.npy")
    gate_up_proj_scales = np.load(PATH + "gate_up_proj_scales.npy")

    down_proj_bias = np.load(PATH + "down_proj_bias.npy")
    down_proj_blocks = np.load(PATH + "down_proj_blocks.npy")
    down_proj_scales = np.load(PATH + "down_proj_scales.npy")

    # (128, 5760)
    # (128, 5760, 90, 16)
    # (128, 5760, 90)
    # print(gate_up_proj_bias.shape)
    # print(gate_up_proj_blocks.shape)
    # print(gate_up_proj_scales.shape)

    # (128, 2880)
    # (128, 2880, 90, 16)
    # (128, 2880, 90)
    # print(down_proj_bias.shape)
    # print(down_proj_blocks.shape)
    # print(down_proj_scales.shape)

    shared_experts = down_proj_blocks.shape[0]
    hidden_size = down_proj_blocks.shape[1]
    batch_size = 1

    batch_inputs = np.random.uniform(low=-1, high=1, size=(batch_size, hidden_size)).astype(np.float16)

    choosed_experts = 1

    out = ffn_layer_mxfp4_gpu(
        gate_up_proj_blocks[choosed_experts],
        gate_up_proj_scales[choosed_experts],
        gate_up_proj_bias[choosed_experts],
        down_proj_blocks[choosed_experts],
        down_proj_scales[choosed_experts],
        down_proj_bias[choosed_experts],
        batch_inputs,
    )

    print(out.shape)
    print(out)
