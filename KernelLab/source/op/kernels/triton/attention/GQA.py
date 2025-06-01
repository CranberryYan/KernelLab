import math
import triton
import triton.language as tl
import torch
import torch.nn as nn
import torch.nn.functional as F

# GQA: 针对Q做了分组处理, 用于加速和降低显存需求
# 核心思想:
#   Q分为多个组, 每组包含若干连续的Q
#   每组内Q只和对应组内的K, V做Attention, 减少全局计算
class GQA(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    # Q   : [B, QH, S, D]
    # K, V: [B, KH, S, D]
    B, QH, S, D = Q.shape
    _, KH, _, _ = K.shape
    self.group_size = QH // KH
    assert QH % self.group_size == 0
    assert QH // self.group_size == KH

    scale = math.sqrt(D)
    output = torch.empty_like(Q)

    for b in range(B):
      for qh in range(QH):
        gid = qh // self.group_size
        q = Q[b, qh, :, :]
        k = K[b, gid, :, :]
        v = V[b, gid, :, :]
        attn_scores = torch.matmul(q, k.T) / scale
        attn_prob = F.softmax(attn_scores, dim=-1)
        output[b, qh] = torch.matmul(attn_prob, v)

    return output

@triton.jit
def GQA_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
               B, QH, KH, S, D,
               group_size,
               scale: tl.constexpr,
               BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr):
  pid_b = tl.program_id(0)
  pid_h = tl.program_id(1)

  group_id = pid_h // group_size

  # Q: [pid_b, pid_h, 0:BLOCK_M, 0:BLOCK_N]
  q_ptrs = q_ptr + \
           pid_b * QH * S * D + \
           pid_h * S * D + \
           tl.arange(0, BLOCK_M)[:, None] * D + \
           tl.arange(0, BLOCK_N)[None, :]
  o_ptrs = o_ptr + \
           pid_b * QH * S * D + \
           pid_h * S * D + \
           tl.arange(0, BLOCK_M)[:, None] * D + \
           tl.arange(0, BLOCK_N)[None, :]

  # K: [pid_b, group_id, 0:BLOCK_M, 0:BLOCK_N]
  k_ptrs = k_ptr + \
           pid_b * KH * S * D + \
           group_id * S * D + \
           tl.arange(0, BLOCK_M)[:, None] * D + \
           tl.arange(0, BLOCK_N)[None, :]
  v_ptrs = v_ptr + \
           pid_b * KH * S * D + \
           group_id * S * D + \
           tl.arange(0, BLOCK_M)[:, None] * D + \
           tl.arange(0, BLOCK_N)[None, :]

  q = tl.load(q_ptrs)
  k = tl.load(k_ptrs)
  v = tl.load(v_ptrs)

  attn_scores = tl.dot(q, tl.trans(k)) * scale
  max = tl.max(attn_scores, axis=1)
  attn_scores = tl.exp(attn_scores - max[:, None])
  sum = tl.sum(attn_scores, axis=1)

  attn_probs = attn_scores / sum[:, None]
  out = tl.dot(attn_probs, v)

  tl.store(o_ptrs, out)

def GQA_host(Q: torch.Tensor,
             K: torch.Tensor,
             V: torch.Tensor,
             group_size: int):
  B, QH, S, D = Q.shape
  _, KH, _, _ = K.shape
  assert QH % KH == 0 and QH == KH * group_size

  output = torch.empty_like(Q)

  BLOCK_M = S
  BLOCK_N = D

  grid = (B, QH, 1)

  GQA_kernel[grid](Q, K, V, output,
                   B, QH, KH, S, D,
                   group_size,
                   scale=1.0 / math.sqrt(D),
                   BLOCK_M=BLOCK_M,
                   BLOCK_N=BLOCK_N,
                   num_warps=4)

  return output

def test_gqa_triton():
  B, QH, KH, S, D = 1, 8, 4, 128, 64
  group_size = QH // KH
  torch.manual_seed(0)

  Q = torch.randn((B, QH, S, D), device='cuda', dtype=torch.float32)
  K = torch.randn((B, KH, S, D), device='cuda', dtype=torch.float32)
  V = torch.randn((B, KH, S, D), device='cuda', dtype=torch.float32)

  ref_out = torch.empty_like(Q)
  for b in range(B):
    for qh in range(QH):
      gid = qh // group_size
      q = Q[b, qh]  # [S, D]
      k = K[b, gid] # [S, D]
      v = V[b, gid] # [S, D]
      attn = (q @ k.T) / math.sqrt(D)  # [S, S]
      prob = attn.softmax(dim=-1)
      ref_out[b, qh] = prob @ v

  res = GQA_host(Q, K, V, group_size)
  GQA_ref = GQA()
  ref = GQA_ref(Q, K, V)

  max_diff = (ref - res).abs().max()
  print(f"Max diff: {max_diff}")
  assert torch.allclose(ref, res, atol=1e-2), "GQA Triton failed"
  print("✅ Triton GQA passed!")

if __name__ == "__main__":
  test_gqa_triton()
