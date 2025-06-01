import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# 模拟实现MHA, 假设QKV已经生成
class MHA(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,
              Q: torch.Tensor,
              K: torch.Tensor,
              V: torch.Tensor,
              scale,
              mask=None):
    # Q: [B, num_heads, seq_len, head_dim]
    # K: [B, num_heads, seq_len, head_dim]
    # V: [B, num_heads, seq_len, head_dim]
    # mask: [B, num_heads, seq_len, head_dim]
    # scale: scalar
    attn_score = torch.matmul(Q, K.transpose(-2, -1)) / scale
    if mask is not None:
      attn_score = attn_score.masked_fill(mask==0, float('-inf'))
    attn_weights = F.softmax(attn_score, dim=-1)
    out = torch.matmul(attn_weights, V)

    return out

# grid = lambda META: (B * QH,
#                      triton.cdiv(S, META["BLOCK_M_SIZE"]),
#                      1)
@triton.jit
def FLA_kernel(q_ptr, k_ptr, v_ptr, output_ptr,
            q_batch_stride, q_heads_stride, q_seq_stride, q_dim_stride,
            k_batch_stride, k_heads_stride, k_seq_stride, k_dim_stride,
            v_batch_stride, v_heads_stride, v_seq_stride, v_dim_stride,
            out_batch_stride, out_heads_stride, out_seq_stride, out_dim_stride,
            num_kv_groups, heads, m_size, n_size,
            scale, causal_mask,
            BLOCK_DHEAD_SIZE: tl.constexpr,
            BLOCK_M_SIZE: tl.constexpr,
            BLOCK_N_SIZE: tl.constexpr):
  head_idx = tl.program_id(axis=0)          # 全局的head_id, 不区分batch
  block_m_idx = tl.program_id(axis=1)       # seq_len维度上的tile
  batch_idx = head_idx // heads             # 全局的batch_id
  q_head_idx = head_idx % heads             # 当前batch中的head_id
  kv_head_idx = q_head_idx // num_kv_groups # GQA -> kv_head_id

  # 注: 此处的m, n不是[M, N], 而是Q, K的seq_len
  # m_offset: [BLOCK_M_SIZE], 并行, 所以有block_m_idx偏移
  # n_offset: [BLOCK_N_SIZE], 循环, online sotfmax
  # head_offset[BLOCK_DHEAD_SIZE]
  m_offset = block_m_idx * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
  n_offset = tl.arange(0, BLOCK_N_SIZE)
  head_offset = tl.arange(0, BLOCK_DHEAD_SIZE)

  # Q: [batch_size, q_head_num, seq_len, head_dim]
  # K: [batch_size, k_head_num, seq_len, head_dim]
  # q_tile: [BLOCK_M_SIZE, BLOCK_DHEAD_SIZE]
  # k_tile: [BLOCK_N_SIZE, BLOCK_DHEAD_SIZE]
  q_offset = batch_idx * q_batch_stride + \
             q_head_idx * q_heads_stride + \
             m_offset[:, None] * q_seq_stride + \
             head_offset[None, :] * q_dim_stride
  k_offset = batch_idx * v_batch_stride + \
             kv_head_idx * v_heads_stride + \
             n_offset[:, None] * v_seq_stride + \
             head_offset[None, :] * v_dim_stride
  v_offset = batch_idx * k_batch_stride + \
             kv_head_idx * k_heads_stride + \
             n_offset[:, None] * k_seq_stride + \
             head_offset[None, :] * k_dim_stride
  o_offset = batch_idx * out_batch_stride + \
             q_head_idx * out_heads_stride + \
             m_offset[:, None] * out_seq_stride + \
             head_offset[None, :] * out_dim_stride

  q_mask = m_offset[:, None] < m_size
  q_tile = tl.load(q_ptr + q_offset, mask=q_mask, other=0.0)

  # online Softmax
  # d_n = d_n-1 * e^(m_n-1 - m_n) + e^(x_n-1 - m_n)
  m_i = tl.zeros((BLOCK_M_SIZE, BLOCK_M_SIZE), dtype=tl.float32) - float("inf")
  d_i = tl.zeros((BLOCK_M_SIZE, BLOCK_M_SIZE), dtype=tl.float32)
  acc_tile = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)

  for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE):
    block_n_offset = block_n_start_idx + n_offset
    # k_mask: [BLOCK_N_SIZE, 1]
    # k_offset: [BLOCK_N_SIZE, BLOCK_DHEAD_SIZE]
    # block_n_offset: [BLOCK_N_SIZE] -> [1, BLOCK_N_SIZE]
    # k_tile: [BLOCK_N_SIZE, BLOCK_DHEAD_SIZE]
    # attn_scores: [BLOCK_M_SIZE, BLOCK_N_SIZE]
    k_mask = block_n_offset[:, None] < n_size
    k_tile = tl.load(k_ptr + k_offset + block_n_offset * k_seq_stride,
                     mask=k_mask, other=0.0)
    attn_scores = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
    attn_scores = tl.dot(q_tile, tl.trans(k_tile))
    v_tile = tl.load(v_ptr + v_offset + block_n_start_idx * v_seq_stride,
                     mask=k_mask, other=0.0).to(tl.float32)

    if causal_mask:
      # mask: [BLOCK_M_SIZE, BLOCK_N_SIZE]
      mask = m_offset[:, None] >= block_n_offset[None, :]
      attn_scores = tl.where(mask, attn_scores * scale, float('-inf'))
    else:
      attn_scores = attn_scores * scale

    m_curr = tl.max(attn_scores, axis=1)[:, None]  # [BLOCK_M]
    m_j = tl.maximum(m_i, m_curr)

    # 每次只处理一个元素, 循环累加
    #   sum, max都是标量, 循环cols次 -> 求出当前行的sum
    #   expf(shared_cols[c] - max_tmp) -> cols次累加
    # if (tid == 0) {
    #   for (int c = 0; c < cols; ++c) {
    #     max_tmp = max(max_tmp, shared_cols[c]);
    #     sum =
    #       sum * expf(pre_max_tmp - max_tmp) +
    #         expf(shared_cols[c] - max_tmp);
    #     pre_max_tmp = max_tmp;
    #   }
    # }

    # 一次处理一个tile, 所以前半部分, 后半部分要一次加完
    #   前半部分: m_i和m_j都有着n行的每行最大值, 且shape均为tile,
    #   后半部分: 没有cols次循环 -> 一次出结果 -> sum
    # d_j = d_i * e^(m_i - m_j) + e^(x_j - m_j)
    d_j = d_i * tl.exp(m_i - m_j) + \
          tl.sum(tl.exp(attn_scores - m_j), axis=1)[:, None]

    # p: (e^(x_j - m_j) / d_j)
    p = tl.exp(attn_scores - m_j) / d_j

    # O_j = O_i * d_i * e^(m_i - m_j) / d_j + (e^(x_j - m_j) / d_j) * V_j
    acc_tile = acc_tile * ((d_i * tl.exp(m_i - m_j)) / d_j) + tl.dot(p, v_tile)

    m_i = m_j
    d_i = d_j

  out_mask = m_offset[:, None] < m_size
  tl.store(output_ptr + o_offset, acc_tile, mask=out_mask)

def FLA_host(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
  output = torch.empty_like(Q)
  assert Q.shape[-1] == K.shape[-1] == V.shape[-1]
  assert (
    Q.dtype == K.dtype == V.dtype == output.dtype
  ), f"All tensors must have the same dtype: \
    {Q.dtype}, {K.dtype}, {V.dtype}, {output.dtype}"

  B, QH, QS, D = Q.shape
  _, KH, KS, _ = K.shape
  num_kv_groups = QH // KH # num_q_heads // num_k_heads
  causal_mask = True
  scale = math.sqrt(D)

  HEAD_DIM = D
  BLOCK_M_SIZE = 32
  BLOCK_N_SIZE = 32
  grid = lambda META: (B * QH,
                       triton.cdiv(QS, META["BLOCK_M_SIZE"]),
                       1)
  FLA_kernel[grid](Q, K, V, output,
                   *Q.stride(),
                   *K.stride(),
                   *V.stride(),
                   *output.stride(),
                   num_kv_groups, QH, QS, KS,
                   scale, causal_mask,
                   HEAD_DIM,
                   BLOCK_M_SIZE,
                   BLOCK_N_SIZE)

  return output

# Prefill阶段 -> 处理完整序列
def test_prefill_stage():
  batch_size = 2
  num_heads = 8
  seq_len = 128
  head_dim = 32

  torch.manual_seed(0)
  Q = torch.rand((batch_size, num_heads, seq_len, head_dim),
                 dtype=torch.float16, device='cuda')
  K = torch.rand((batch_size, num_heads, seq_len, head_dim),
                 dtype=torch.float16, device='cuda')
  V = torch.rand((batch_size, num_heads, seq_len, head_dim),
                 dtype=torch.float16, device='cuda')
  mask = torch.tril(
    torch.ones((seq_len, seq_len), device='cuda')).view(1, 1, seq_len, seq_len)
  scale = math.sqrt(head_dim)

  res = FLA_host(Q, K, V)
  mha = MHA()
  ref = mha(Q, K, V, scale, mask=mask)

  if torch.allclose(res, ref, atol=1e-2):
    print(
      "Prefill Stage Test Passed: \
        Triton output matches PyTorch standard implementation.")
  else:
    max_diff = (res - ref).abs().max()
    print(f"Prefill Stage Test Failed: Maximum difference {max_diff}")
  for i in range(0, 32, 1):
    print(f"result = {res[0][0][0][i]}, ref = {ref[0][0][0][i]}")

if __name__ == "__main__":
  print(" ================== Running Prefill Stage Test")
  test_prefill_stage()
