import torch

# Q [n_tokens, n_heads, q_mult, d_head]
# K [n_tokens, n_heads, d_head]
# V [n_tokens, n_heads, d_head]
# S [n_heads, q_mult]

def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    
    # K,V [n_tokens, n_heads, d_head] => [n_tokens, n_heads, q_mult, d_head]
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    # S [n_heads, q_mult] => [n_heads, q_mult, n_tokens, 1]
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    
    # 上三角 MASK
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    # sliding_window 下三角
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )

    # [n_heads, q_mult, n_tokens, n_tokens]
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    
    # [n_heads, q_mult, n_tokens, n_tokens] + [n_heads, q_mult, n_tokens, 1] => [n_heads, q_mult, n_tokens, n_tokens+1]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    # [n_heads, q_mult, n_tokens, n_tokens+1] => [n_heads, q_mult, n_tokens, n_tokens]
    W = W[..., :-1]

    # [n_tokens, n_heads, q_mult, d_head]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


# ------------------ 测试 ------------------
torch.manual_seed(0)

n_tokens = 4
n_heads = 2
q_mult = 1
d_head = 8

Q = torch.randn(n_tokens, n_heads, q_mult, d_head)
K = torch.randn(n_tokens, n_heads, d_head)
V = torch.randn(n_tokens, n_heads, d_head)
S = torch.randn(n_heads, q_mult)
sm_scale = 1.0 / (d_head ** 0.5)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("S shape:", S.shape)

out = sdpa(Q, K, V, S, sm_scale, sliding_window=2)
print("Output shape:", out.shape)
print("Output:", out)
