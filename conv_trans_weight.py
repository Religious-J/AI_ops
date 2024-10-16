import numpy as np

# (Cin, kH, kW, Cout) =>  (kH, kW, (Cin+31)//32, (Cout+31)//32, 32, 32)
def conv_pvc_trans_weight_cpu(ptr_in, Cin, kH, kW, Cout):
    # 按 32 分块
    Cin_tot = (Cin + 31) // 32
    Cout_tot = (Cout + 31) // 32
    
    # 初始化输出矩阵, 大小为 (H * W * Cin/32 * Cout/32 * 32 * 32)
    ptr_out = np.zeros((kH * kW, Cin_tot, Cout_tot, 32, 32), dtype=np.float32)

    tot_mission = Cin_tot * kH * kW * Cout_tot
    blk = Cin_tot * Cout_tot

    for idx in range(tot_mission):
        wh_idx = idx // blk
        Cio_idx = idx % blk
        Ci_idx = Cio_idx // Cout_tot
        Co_idx = Cio_idx % Cout_tot

        offset_out_col = Co_idx * 32
        offset_in_row = Ci_idx * 32

        func_data_col = min(32, Cout - offset_out_col)
        func_data_row = min(32, Cin - offset_in_row)

        for h in range(kH):
            for w in range(kW):
                for ci in range(func_data_row):
                    input_idx = offset_in_row + ci
                    input_offset = (input_idx * kH * kW * Cout) + (wh_idx * Cout) + offset_out_col
                    output_idx = (wh_idx * kW + w) * Cin_tot * Cout_tot * 32 * 32 + Ci_idx * Cout_tot * 32 * 32 + Co_idx * 32 * 32
                    ptr_out[w * kH + h, Ci_idx, Co_idx, ci, :func_data_col] = ptr_in[input_offset:input_offset + func_data_col]
    
    return ptr_out

def main():
    Cin = 64
    Cout = 128
    kH = 3
    kW = 3
    input_shape = (Cin, kH, kW, Cout)
    ptr_in = np.random.rand(Cin, kH, kW, Cout).astype(np.float32)
    ptr_in_flat = ptr_in.flatten()
    ptr_out = conv_pvc_trans_weight_cpu(ptr_in_flat, Cin, kH, kW, Cout)
    
    print("Output shape:", ptr_out.shape)
    print("First element of output block:", ptr_out[0, 0, 0])

if __name__ == '__main__':
    main()
