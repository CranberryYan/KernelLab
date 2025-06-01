import math
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd
import torch.nn.functional as F
import triton
from triton import Config
import triton.language as tl
import matplotlib.pyplot as plt

def MHA(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
  batch_size, seq_len, head_num, head_dim = Q.shape

  mask = 1.0 - torch.tril(
    torch.ones((seq_len, seq_len), device=Q.device), diagonal=0)
  mask = mask.view(1, 1, seq_len, seq_len)
  mask = mask.masked_fill(mask.to(torch.bool), float('-1e9'))

  q = Q.transpose(1, 2)
  k = K.transpose(1, 2)
  v = V.transpose(1, 2)

  attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
  attn_probs = F.softmax(attn_scores.float() + mask, dim=-1).to(Q.dtype)
  output = torch.matmul(attn_probs, v)\
                .transpose(1, 2)\
                .contiguous()\
                .reshape(batch_size, seq_len, head_num, head_dim)

  return output

def sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
  batch_size, seq_len, head_num, head_dim = Q.shape
  q = Q.transpose(1, 2)
  k = K.transpose(1, 2)
  v = V.transpose(1, 2)
  output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
  output = output.transpose(1, 2)\
                 .contiguous()\
                 .reshape(batch_size, seq_len, head_num, head_dim)
  return output

def prefill_stage_golden(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                       b_start_loc, b_seq_len, use_sdpa=False):
  output = torch.empty_like(Q)
  Z = b_start_loc.shape[0]
  for i in range(Z):
    start = b_start_loc[i]
    end = start + b_seq_len[i]
    q = Q[start: end].unsqueeze(0)
    k = K[start: end].unsqueeze(0)
    v = V[start: end].unsqueeze(0)
    if use_sdpa:
      o = sdpa(q, k, v)
    else:
      o = MHA(q, k, v)
    output[start: end] = o.unsqueeze(0)

  return output

attention_configs = [
  Config(
    {"BLOCK_M_SIZE": block_m, "BLOCK_N_SIZE": block_n},
    num_stages=num_stages,
    num_warps=8,
  )
  for block_m in [16, 32, 64]
  for block_n in [32, 64, 128]
  for num_stages in [3, 4, 5, 6]
]

@triton.autotune(
  configs=attention_configs, key=["BLOCK_DHEAD_SIZE", "heads", "num_kv_groups"])
@triton.jit
def FLA_v2_no_padding_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                             b_start_loc, b_seq_len,
                             scale, heads, num_kv_groups,
                             stride_q_bs, stride_q_heads, stride_q_dim,
                             stride_k_bs, stride_k_heads, stride_k_dim,
                             stride_v_bs, stride_v_heads, stride_v_dim,
                             stride_o_bs, stride_o_heads, stride_o_dim,
                             BLOCK_DHEAD_SIZE: tl.constexpr,
                             BLOCK_M_SIZE: tl.constexpr,
                             BLOCK_N_SIZE: tl.constexpr):
  # shape: [batch * max_seq_len, head_num, head_dim]
  # grid = (triton.cdiv(max_seq_len, BLOCK_M_SIZE), batch_size * HEAD_NUM, 1)
  # 每个head都是独立的
  batch_idx = tl.program_id(1) // heads
  block_m_idx = tl.program_id(0)
  head_idx = tl.program_id(1) % heads
  kv_head_idx = head_idx // num_kv_groups
  block_start = block_m_idx * BLOCK_M_SIZE

  # # 代表4个batch
  # b_seq_len = torch.tensor([512, 1024, 512, 1024],
  #                          dtype=torch.int32, device="cuda")
  # # 每个batch的起始位置
  # b_start_loc = torch.tensor([0, 512, 1536, 2048],
  #                            dtype=torch.int32, device="cuda")
  cur_seq_len = tl.load(b_seq_len + batch_idx)
  cur_seq_start_loc = tl.load(b_start_loc + batch_idx)

  # 同FLA_v1, 此处的M, N不是[M, N]而是Q, K分别的seq_len维度的偏移
  m_offset = block_start + tl.arange(0, BLOCK_M_SIZE)
  n_offset = tl.arange(0, BLOCK_N_SIZE)
  h_offset = tl.arange(0, BLOCK_DHEAD_SIZE)

  # 每个head是独立的, 当前block循环处理当前head的seq_len维度
  # shape: [batch * max_seq_len, head_num, head_dim]
  # q_offset: [BLOCK_M_SIZE, BLOCK_DHEAD_SIZE]
  # stride_q_bs: head_num * head_dim
  # stride_q_heads: head_dim
  # stride_q_dim: 1
  q_offset = (cur_seq_start_loc + m_offset)[:, None] * stride_q_bs +\
             head_idx * stride_q_heads +\
             h_offset[None, :] * stride_q_dim
  o_offset = (cur_seq_start_loc + m_offset[:, None]) * stride_o_bs +\
             head_idx * stride_o_heads +\
             h_offset[None, :] * stride_o_dim
  q_tile = tl.load(q_ptr + q_offset,
                   mask=m_offset[:, None] < cur_seq_len,
                   other=0.0)

  # 同FLA_v1, Q行并行, 所以m_offset与pid有关
  #   K, V循环, 所以n_offset与pid无关但在下面的loop会不断自增
  k_offset = (cur_seq_start_loc + n_offset)[None, :] * stride_k_bs +\
             kv_head_idx * stride_k_heads +\
             h_offset[:, None] * stride_k_dim
  v_offset = (cur_seq_start_loc + n_offset)[:, None] * stride_v_bs +\
             kv_head_idx * stride_v_heads +\
             h_offset[None, :] * stride_v_dim

  k_ptrs = k_ptr + k_offset
  v_ptrs = v_ptr + v_offset
  out_ptrs = o_ptr + o_offset

  m_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
  d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
  acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)

  # 处理不定长(kv_seq_len -> n_size)的数据(不同batch的长度不同)
  # tl.where: tensor版本的三目运算符
  block_mask = tl.where(block_start < cur_seq_len, 1, 0)
  block_end = tl.minimum(block_start + BLOCK_M_SIZE, cur_seq_len)

  for start in range(0, block_mask * block_end, BLOCK_N_SIZE):
    # 优化指令: 告诉编译器, start一定是BLOCK_N_SIZE的整数倍
    start = tl.multiple_of(start, BLOCK_N_SIZE)

    # load k_tile and v_tile
    # m_offset = block_start + tl.arange(0, BLOCK_M_SIZE)
    # n_offset = tl.arange(0, BLOCK_N_SIZE)
    # h_offset = tl.arange(0, BLOCK_DHEAD_SIZE)
    # k_tile: [BLOCK_DHEAD_SIZE, BLOCK_N_SIZE]
    # v_tile: [BLOCK_M_SIZE, BLOCK_N_SIZE]
    k_tile = tl.load(k_ptrs + start * stride_k_bs,
                     mask=(start + n_offset[None, :]) < block_end,
                     other=0.0)
    v_tile = tl.load(v_ptrs + start * stride_v_bs,
                     mask=(start + n_offset[:, None]) < block_end,
                     other=0.0)
    # attn_scores: [BLOCK_M_SIZE, BLOCK_N_SIZE]
    attn_scores = tl.dot(q_tile, k_tile)

    casual_mask = m_offset[:, None] >= (start + n_offset[None, :])
    attn_scores = tl.where(casual_mask, attn_scores * scale, -1.0e8)

    # d_j = d_i * e^(m_i - m_j) + e^(x_j - m_j)
    m_j = tl.maximum(m_i, tl.max(attn_scores, 1))
    alpha = tl.math.exp2(m_i - m_j)
    beta = tl.math.exp2(attn_scores - m_j[:, None])
    d_j = d_i * alpha + tl.sum(beta, 1)

    # O_j = O_i * d_i * e^(m_i - m_j) / d_j + (e^(x_j - m_j) / d_j) * V_j
    # O_j = (O_i * d_i * e^(m_i - m_j) + (e^(x_j - m_j)) * V_j) / d_j
    beta = beta.to(v_tile.dtype)
    acc = acc * alpha[:, None]  # O_i * d_i * e^(m_i - m_j)
    acc += tl.dot(beta, v_tile) # e^(x_j - m_j)) * V_j

    m_i = m_j
    d_i = d_j
  acc = acc / d_i[:, None]      # acc / d_j
  tl.store(out_ptrs, acc, mask=m_offset[:, None] < cur_seq_len)

@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16)
def FLA_v2_no_padding_host(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           scale, b_start_loc, b_seq_len, max_seq_len):
  # 此处的m, n代表QK各自的seq_len
  # Q: [bs * m_size, head_num, head_dim]
  # K: [bs * n_size, head_num, head_dim]
  # V: [bs * n_size, head_num, head_dim]
  BLOCK_SIZE = 64
  HEAD_NUM, HEAD_DIM = Q.shape[1], Q.shape[2]
  output = torch.empty_like(Q)
  batch_size = b_seq_len.shape[0]
  num_kv_groups = Q.shape[1] // K.shape[1]
  grid = lambda META: (triton.cdiv(max_seq_len, META['BLOCK_M_SIZE']),
                       batch_size * HEAD_NUM,
                       1)
  FLA_v2_no_padding_kernel[grid](Q, K, V, output,
                        b_start_loc, b_seq_len, scale, HEAD_NUM, num_kv_groups,
                        *Q.stride(), *K.stride(), *V.stride(), *output.stride(),
                        BLOCK_DHEAD_SIZE=HEAD_DIM)

  return output

def FLA_v2_no_padding_benchmark(
  use_sdpa,
  batch=4, head_num=32, head_dim=128, max_seq_len_list=[1024, 2048, 4096]):

  # 匹配FLA_v2中的计算, 以2为底, 而不是以e为底
  # softmax = 2^(attn_score - max) / sum(2^(attn_score - max))
  scale = 1 / math.sqrt(head_dim) * 1.4426950408889634
  max_seq_len = max_seq_len_list[0]

  # Q: [batch_size, seq_len, head_num, head_dim] ->
  #   [batch_size * seq_len, head_num, head_dim]
  # 配合FLA_v2_no_padding_host中的
  # Q: [bs * m_size, head_num, head_dim]
  torch.manual_seed(0)
  Q = torch.randn((batch, max_seq_len, head_num, head_dim),
                  dtype=torch.float16, device='cuda')
  Q = Q.view(batch*max_seq_len, head_num, head_dim)
  K = torch.randn((batch, max_seq_len, head_num, head_dim),
                  dtype=torch.float16, device='cuda')
  K = K.view(batch*max_seq_len, head_num, head_dim)
  V = torch.randn((batch, max_seq_len, head_num, head_dim),
                  dtype=torch.float16, device='cuda')
  V = V.view(batch*max_seq_len, head_num, head_dim)

  b_seq_len = torch.tensor([512, 1024, 512, 1024],
                           dtype=torch.int32, device="cuda")
  b_start_loc = torch.tensor([0, 512, 1536, 2048],
                             dtype=torch.int32, device="cuda")

  triton_output = FLA_v2_no_padding_host(Q, K, V, scale,
                                         b_start_loc, b_seq_len, max_seq_len)
  torch_output = prefill_stage_golden(Q, K, V,
                                      b_start_loc, b_seq_len, use_sdpa=use_sdpa)
  print(f"The maximum difference between torch and triton is \
    {torch.max(torch.abs(torch_output - triton_output))}")
  if torch.allclose(triton_output, torch_output, atol=1e-2):
    print(
    "Prefill Stage Test Passed: \
      Triton output matches PyTorch standard implementation.")
  else:
    max_diff = (triton_output - torch_output).abs().max()
    print(f"Prefill Stage Test Failed: Maximum difference {max_diff}")
  for i in range(0, 32, 1):
    print(f"result = {triton_output[0][i][i]}, ref = {torch_output[0][i][i]}")

  # 性能统计
  flash_times = []
  golden_times = []
  iterations = 50
  for seq_len in max_seq_len_list:
    Q = torch.randn((batch, seq_len, head_num, head_dim),
                    dtype=torch.float16, device='cuda')
    Q = Q.view(batch*seq_len, head_num, head_dim)
    K = torch.randn((batch, seq_len, head_num, head_dim),
                    dtype=torch.float16, device='cuda')
    K = K.view(batch*seq_len, head_num, head_dim)
    V = torch.randn((batch, seq_len, head_num, head_dim),
                    dtype=torch.float16, device='cuda')
    V = V.view(batch*seq_len, head_num, head_dim)
    b_start_loc = torch.tensor(
      [0, seq_len, 2 * seq_len, 3 * seq_len], dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch,), seq_len, device='cuda', dtype=torch.int32)

    # warm up
    _ = FLA_v2_no_padding_host(Q, K, V, scale, b_start_loc, b_seq_len, seq_len)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
      _ = FLA_v2_no_padding_host(Q, K, V,
                                 scale, b_start_loc, b_seq_len, seq_len)
    end_event.record()
    torch.cuda.synchronize()
    flash_time = start_event.elapsed_time(end_event) / iterations
    flash_times.append(flash_time)

    # warm up
    _ = prefill_stage_golden(Q, K, V, b_start_loc, b_seq_len, use_sdpa=use_sdpa)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(iterations):
      _ = prefill_stage_golden(Q, K, V,
                               b_start_loc, b_seq_len, use_sdpa=use_sdpa)
    end_event.record()
    torch.cuda.synchronize()
    standard_time = start_event.elapsed_time(end_event) / iterations
    golden_times.append(standard_time)

    print(
      f"max_seq_len = {seq_len:4d}: \
        flash_attn = {flash_time:.3f} ms, \
          standard_attn = {standard_time:.3f} ms")

  plt.figure(figsize=(8, 5))
  plt.plot(max_seq_len_list,
           flash_times,
           marker="o",
           label="FLA_v2")
  plt.plot(max_seq_len_list,
           golden_times,
           marker="s",
           label="Standard Attention")
  plt.xlabel("max_seq_len (kv cache length)")
  plt.ylabel("Average execution time (ms)")
  plt.title("Prefill Stage Performance Comparison")
  plt.legend()
  plt.grid(True)
  plt.savefig(f"./FLA_v2_benchamrk_{use_sdpa}.png")

if __name__ == "__main__":
  stats = FLA_v2_no_padding_benchmark(use_sdpa=False)
  print("Benchmark statistics:", stats)
