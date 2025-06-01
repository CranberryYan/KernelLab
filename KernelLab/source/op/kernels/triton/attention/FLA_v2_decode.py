"""
FLA_v2:
prefill阶段:
  对Q的seq_len进行分块处理(BLOCK_M_SIZE), 在K, V的seq_len进行循环处理(BLOCK_N_SIZE)
decode阶段:
  问题:
    1. 维度不平衡: 在decode阶段, Q的seq_len为1, 而K, V则包含所有的历史信息且随着上下文不断增加
    2. 并行度不足: prefill阶段会在seq_len维度进行分块, 但是decode阶段无法分块
  解决:
    1. 序列分区: 将K, V的序列长度分为多个分区, 每次分区再细分为多个block
    2. 并行计算: Q与K, V的对应分区并行计算
    3. 中间结果管理: 每个分区的计算结果存储再O_MID中
  即使seq_len为1, 通过K, V序列维度的并行计算提高GPU利用率
  
  stage1: 对分区内所有块的注意力分数进行计算, 这个过程是逐个处理区内的每个块来完成的,
  每处理完一个分区内的所有块就更新该分区的中间输出结果
  stage2: 所有分区计算完成后, 根据各分区中间结果的最大值, 归一化项(即分区内的和)以及分块内的中间输出, 计算全局输出,
  生成完整的注意力机制输出结果
"""

import math
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd
import torch.nn.functional as F
import triton
import triton.language as tl
import matplotlib.pyplot as plt

def MHA(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
  head_dim = Q.shape[-1]

  q = Q.transpose(0, 1) # [head_num, 1, head_dim]
  k = K.transpose(0, 1) # [head_num, seq_len, head_dim]
  v = V.transpose(0, 1) # [head_num, seq_len, head_dim]

  attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
  attn_probs = F.softmax(attn_scores.float(), dim=-1).to(Q.dtype)
  output = torch.matmul(attn_probs, v)\
                .transpose(0, 1)\
                .contiguous()

  return output

def torch_attention_with_kvcache(Q: torch.Tensor,
                                 K_cache: torch.Tensor,
                                 V_cache: torch.Tensor,
                                 b_start_loc, b_seq_len):
  out = torch.empty_like(Q)
  Z = Q.shape[0]
  for i in range(Z):
    start = b_start_loc[i]
    end = start + b_seq_len[i]
    q = Q[i: i + 1]  # (1, head_num, head_dim)
    k = K_cache[start: end]  # (seqlen, head_num, head_dim)
    v = V_cache[start: end]  # (seqlen, head_num, head_dim)
    o = MHA(q, k, v)
    out[i : i + 1] = o
  return out

# eg:
# Q: [batch, 1, head_num, head_dim]
# K: [batch, seq_len, head_num, head_dim], seq_len: 128(历史token数量)
# attn_scores:
#   [batch, head_num, 1, head_dim] * [batch, head_num, head_dim, seq_len] =
#   [batch, head_num, 1, seq_len] * [batch, head_num, seq_len, head_num] =
#   [batch, head_num, 1, head_dim] -> [batch, 1, head_num, head_dim]
# partition: 64个token为一组(BLOCK_SEQ)
# blocks: 16个token为一组(BLOCK_N_SIZE)
# 并行: Q只有一个token, 但是与partition并行计算
# 会生成part个中间结果
@triton.jit
def flash_decode_stage1_kernel(Q, K, V, scale,
  b_req_tokens_table: torch.Tensor,
  b_seqlen,
  num_kv_groups,
  mid_o, mid_o_logExpSum,
  stride_req_to_tokens_b,
  q_bs_stride, q_heads_stride, q_dim_stride,
  k_bs_stride, k_heads_stride, k_dim_stride,
  v_bs_stride, v_heads_stride, v_dim_stride,
  mido_batch_stride, mido_heads_stride, mido_partitions_stride, mido_dim_stride,
  mido_les_batch_stride, mido_les_heads_stride, mido_les_partitions_stride,
  BLOCK_SEQ: tl.constexpr,
  BLOCK_N_SIZE: tl.constexpr,
  BLOCK_DHEAD_SIZE: tl.constexpr):
  # grid = (batchs,
  #         num_heads,
  #         triton.cdiv(max_actual_seq_len+PARTITION_SIZE-1, PARTITION_SIZE))
  batch_id = tl.program_id(0)
  head_id = tl.program_id(1)
  seq_block_id = tl.program_id(2)
  kv_head_id = head_id // num_kv_groups

  # b_seqlen: [batch] -> [tokens, tokens, tokens, tokens]
  # 将K的seq_len分为多个part, 每个part再分为多个block
  cur_batch_seq_len = tl.load(b_seqlen + batch_id)
  cur_batch_partition_start = seq_block_id * BLOCK_SEQ
  cur_batch_partition_end = tl.minimum(cur_batch_seq_len,
                                       cur_batch_partition_start + BLOCK_SEQ)
  cur_batch_partition_len = tl.where(
    cur_batch_partition_end - cur_batch_partition_start > 0,
    cur_batch_partition_end - cur_batch_partition_start,
    0)
  num_blocks = tl.cdiv(cur_batch_partition_len, BLOCK_N_SIZE)

  # m_offset: 没有m_offset, Q的seq_len为1
  # n_offset: [BLOCK_N_SIZE]
  n_offset = cur_batch_partition_start + tl.arange(0, BLOCK_N_SIZE)
  h_offset = tl.arange(0, BLOCK_DHEAD_SIZE)

  # Q: [batch, 1, head_num, head_dim]
  q_offset = batch_id * q_bs_stride + \
             head_id * q_heads_stride + \
             h_offset * q_dim_stride

  # q_ptrs: [BLOCK_DHEAD_SIZE]
  q_ptrs = Q + q_offset
  # q_tile: [BLOCK_DHEAD_SIZE]
  q_tile = tl.load(q_ptrs)

  # K: [batch, max_seq_len, head_num, head_dim]
  k_offset = kv_head_id * k_heads_stride + \
             h_offset[None, :] * k_dim_stride

  # 归一化项和acc
  d_i = 0.0
  m_i = -float("inf")
  acc = tl.zeros([BLOCK_DHEAD_SIZE], dtype=tl.float32)
  for start in range(0, num_blocks, 1):
    # n_offset_: [BLOCK_N_SIZE]
    n_offset_ = n_offset + start * BLOCK_N_SIZE
    mask = n_offset_ < cur_batch_partition_end
    # b_req_tokens_table: tensor [batch, tokens]
    # b_req_tokens_table + batch_id * stride_req_to_tokens_b + n_offset_
    #   相当于 b_req_tokens_table[batch_id, n_offset_] 是一个scalar
    # stride_req_to_tokens_b: tokens
    # k_loc: [BLOCK_N_SIZE]
    k_loc = tl.load(b_req_tokens_table + \
                    batch_id * stride_req_to_tokens_b + n_offset_,
                    mask,
                    other=0.0)
    # k_ptrs: [BLOCK_N_SIZE, 1]
    k_ptrs = k_loc[:, None] * k_bs_stride + k_offset

    # 注: ptr与mask的dim要一致
    k_tile = tl.load(K + k_ptrs, mask[:, None], other=0.0)
    v_tile = tl.load(V + k_ptrs, mask[:, None], other=0.0)

    # q_tile: [BLOCK_DHEAD_SIZE] -> [1, BLOCK_DHEAD_SIZE]
    # k_tile: [BLOCK_N_SIZE, 1]
    # attn_scores: [BLOCK_DHEAD_SIZE, BLOCK_N_SIZE] -> [1, BLOCK_N_SIZE]
    # 计算 attn_scores^T
    attn_scores = tl.sum(q_tile[None, :] * k_tile, axis=1)  # [BLOCK_N]
    attn_scores *= scale
    attn_scores = tl.where(mask, attn_scores, float("-inf"))  # [BLOCK_N]

    m_j = tl.maximum(m_i, tl.max(attn_scores))  # 标量
    beta = tl.exp(attn_scores - m_j)  # [BLOCK_N]

    alpha = tl.exp(m_i - m_j)
    d_j = alpha * d_i + tl.sum(beta, axis=0)

    acc = alpha * acc + tl.sum(beta[:, None] * v_tile, axis=0)  # [BLOCK_DMODEL]

    m_i = m_j
    d_i = d_j
  mid_o_offset = batch_id * mido_batch_stride + \
                 head_id * mido_heads_stride + \
                 seq_block_id * mido_partitions_stride + \
                 h_offset * mido_dim_stride
  mid_o_les_offset = batch_id * mido_les_batch_stride + \
                     head_id * mido_les_heads_stride + \
                     seq_block_id * mido_les_partitions_stride

  need_store = tl.where(num_blocks == 0, 0, 1)
  for _ in range(0, need_store, 1):
    tl.store(mid_o + mid_o_offset, acc / d_i)
    tl.store(mid_o_logExpSum + mid_o_les_offset, m_i + tl.log(d_i))

@triton.jit
def flash_decode_stage2_kernel(mid_o, mid_o_logExpSum, output,
  mido_batch_stride, mido_heads_stride, mido_partitions_stride, mido_dim_stride,
  mido_les_batch_stride, mido_les_heads_stride, mido_les_partitions_stride,
  o_bs_stride, o_heads_stride, o_dim_stride, b_seqlen,
  BLOCK_DHEAD_SIZE: tl.constexpr,
  BLOCK_SEQ: tl.constexpr):
  batch_id = tl.program_id(0)
  head_id = tl.program_id(1)
  cur_batch_seq_len = tl.load(b_seqlen + batch_id)
  h_offset = tl.arange(0, BLOCK_DHEAD_SIZE)
  mid_o_part_offset = batch_id * mido_batch_stride + \
                      head_id * mido_heads_stride + \
                      h_offset * mido_dim_stride
  max_part_offset = batch_id * mido_les_batch_stride + \
                    head_id * mido_les_heads_stride

  mid_o_part_ptrs = mid_o + mid_o_part_offset
  max_part_ptrs = mid_o_logExpSum + max_part_offset

  d_i = 0.0
  m_i = -float('inf')
  acc = tl.zeros([BLOCK_DHEAD_SIZE],dtype=tl.float32)

  num_partitions = tl.cdiv(cur_batch_seq_len, BLOCK_SEQ)
  for block_seq_n in range(0, num_partitions, 1):
    mid_o_part = tl.load(mid_o_part_ptrs + block_seq_n * mido_partitions_stride)
    max_part = tl.load(max_part_ptrs + block_seq_n * mido_les_partitions_stride)
    m_j = tl.maximum(m_i, max_part)
    alpha = tl.exp(m_i - m_j)
    beta = tl.exp(max_part - m_j)
    d_j = alpha * d_i + beta
    acc = alpha * acc + beta * mid_o_part

    m_i = m_j
    d_i = d_j

  out_offset = batch_id * o_bs_stride + \
               head_id * o_heads_stride + \
               h_offset * o_dim_stride
  tl.store(output + out_offset, acc / d_i)

@torch.no_grad()
def flash_decode_stage1(Q: torch.Tensor,
                        K: torch.Tensor,
                        V: torch.Tensor,
                    scale, b_req_tokens_table, b_seq_len,
                    max_actual_seq_len, mid_o, mid_o_logexpsum, PARTITION_SIZE):
  BLOCK_N_SIZE = 16 # 一次从KV_Cache中加载的token的个数

  assert (PARTITION_SIZE % BLOCK_N_SIZE == 0), \
    "PARTITION_SIZE 必须是 BLOCK_N_SIZE 的倍数"

  batchs, num_heads, head_dim = Q.shape

  grid = (batchs, num_heads,
          triton.cdiv(max_actual_seq_len + PARTITION_SIZE - 1, PARTITION_SIZE))
  num_kv_groups = Q.shape[1] // K.shape[1]

  flash_decode_stage1_kernel[grid](Q, K, V, scale,
                                   b_req_tokens_table, b_seq_len, num_kv_groups,
                                   mid_o, mid_o_logexpsum,
                                   b_req_tokens_table.stride(0),
                                   *Q.stride(), *K.stride(), *V.stride(),
                                   *mid_o.stride(), *mid_o_logexpsum.stride(),
                                   BLOCK_SEQ=PARTITION_SIZE,
                                   BLOCK_N_SIZE=BLOCK_N_SIZE,
                                   BLOCK_DHEAD_SIZE=head_dim,
                                   num_warps=2, num_stages=2)

@torch.no_grad()
def flash_decode_stage2(mid_o, mid_o_logexpsum,
                        atten_output, b_seq_len, PARTITION_SIZE):
  batch, head_num, head_dim = mid_o.shape[0], mid_o.shape[1], mid_o.shape[-1]
  grid = (batch, head_num)

  flash_decode_stage2_kernel[grid](mid_o, mid_o_logexpsum,
                                   atten_output,
                                   *mid_o.stride(), *mid_o_logexpsum.stride(),
                                   *atten_output.stride(),
                                   b_seq_len,
                                   BLOCK_DHEAD_SIZE=head_dim,
                                   BLOCK_SEQ=PARTITION_SIZE,
                                   num_warps=4, num_stages=2)

@torch.no_grad()
def flash_decoding(Q: torch.Tensor,
                   K_cache: torch.Tensor,
                   V_cache: torch.Tensor,
                   scale, b_req_tokens_table, b_seq_len, max_actual_seq_len):
  # Q: [batch_size * 1, head_num, head_dim]
  # K_cache: [batch * tokens, kv_num_head, head_dim]
  # V_cache: [batch * tokens, kv_num_head, head_dim]
  PARTITION_SIZE = 64
  batch, head_num, head_dim = Q.shape

  max_num_partitions = \
    (max_actual_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE

  mid_o = torch.empty((batch, head_num, max_num_partitions, head_dim),
                      dtype=torch.float32, device='cuda')
  mid_o_logexpsum = torch.empty((batch, head_num, max_num_partitions),
                                dtype=torch.float32, device='cuda')

  # decode stage 1: attention in partitions
  flash_decode_stage1(Q, K_cache, V_cache, scale, b_req_tokens_table, b_seq_len,
                    max_actual_seq_len, mid_o, mid_o_logexpsum, PARTITION_SIZE)

  # decode stage 2: reduction among partitions
  atten_output = torch.empty_like(Q)

  flash_decode_stage2(mid_o, mid_o_logexpsum, atten_output, b_seq_len, PARTITION_SIZE)

  return atten_output

def plot_performance_comparison(token_sizes, warmup_iterations=10, test_iterations=50):
  device = torch.device("cuda")
  batch = 4
  num_heads = 32
  head_dim = 64
  scale = 1.0 / (head_dim**0.5)
  q = torch.randn(batch * 1, num_heads, head_dim, device=device)

  flash_times = []
  standard_times = []

  for tokens in token_sizes:
    print(f"\ntest token size: {tokens}")
    k_cache = torch.randn(batch * tokens, num_heads, head_dim, device=device)
    v_cache = torch.randn(batch * tokens, num_heads, head_dim, device=device)
    # b_req_tokens_table: [batch, tokens] KV_Cache的最大容量
    b_req_tokens_table = torch.arange(
      0, tokens, device=device, dtype=torch.int32
    ).repeat(batch, 1)
    # b_seq_len: [batch] -> [tokens, tokens, tokens, tokens]
    b_seq_len = torch.full((batch,), tokens, device=device, dtype=torch.int32)
    b_start_loc = torch.tensor([0, tokens, 2 * tokens, 3 * tokens],
                               dtype=torch.int32, device="cuda")
    max_actual_seq_len = tokens

    for _ in range(warmup_iterations):
      _ = flash_decoding(q, k_cache, v_cache,
                              scale,
                              b_req_tokens_table, b_seq_len,
                              max_actual_seq_len)

    torch.cuda.synchronize()
    flash_start = torch.cuda.Event(enable_timing=True)
    flash_end = torch.cuda.Event(enable_timing=True)
    flash_start.record()
    for _ in range(test_iterations):
      _ = flash_decoding(q, k_cache, v_cache,
                              scale,
                              b_req_tokens_table, b_seq_len,
                              max_actual_seq_len)
    flash_end.record()
    torch.cuda.synchronize()
    flash_avg = flash_start.elapsed_time(flash_end) / test_iterations
    flash_times.append(flash_avg)
    print(f"Flash Decoding avg dur: {flash_avg:.3f} ms")

    for _ in range(warmup_iterations):
      _ = torch_attention_with_kvcache(q, k_cache, v_cache,
                                       b_start_loc, b_seq_len)
    torch.cuda.synchronize()
    std_start = torch.cuda.Event(enable_timing=True)
    std_end = torch.cuda.Event(enable_timing=True)
    std_start.record()
    for _ in range(test_iterations):
      _ = torch_attention_with_kvcache(q, k_cache, v_cache,
                                       b_start_loc, b_seq_len)
    std_end.record()
    torch.cuda.synchronize()
    std_avg = std_start.elapsed_time(std_end) / test_iterations
    standard_times.append(std_avg)
    print(f"Standard Attention avg dur: {std_avg:.3f} ms")

  plt.figure(figsize=(8, 6))
  plt.plot(token_sizes, flash_times, marker="o", label="Flash Decoding")
  plt.plot(token_sizes, standard_times, marker="o", label="Standard Attention")
  plt.xlabel("Token Size (kv cache length)")
  plt.ylabel("Average Time (ms)")
  plt.title("Performance Comparison: Flash Decoding vs Standard Attention")
  plt.legend()
  plt.grid(True)
  plt.savefig("./FLA_v3_benchamrk.png")

def FLA_v3_benchmark():
  torch.manual_seed(0)
  batch = 4
  head_num = 32
  head_dim = 64
  max_seq_len = 2048
  scale = 1.0 / math.sqrt(head_dim)

  # 自回归阶段, 每次生成一个token -> Q: [batch, 1, head_num, head_dim]
  torch.manual_seed(0)
  Q = torch.randn((batch, 1, head_num, head_dim),
                  dtype=torch.float16, device='cuda')
  Q = Q.view(batch*1, head_num, head_dim)
  K_cache = torch.randn((batch, max_seq_len, head_num, head_dim),
                  dtype=torch.float16, device='cuda')
  K_cache = K_cache.view(batch*max_seq_len, head_num, head_dim)
  V_cache = torch.randn((batch, max_seq_len, head_num, head_dim),
                  dtype=torch.float16, device='cuda')
  V_cache = V_cache.view(batch*max_seq_len, head_num, head_dim)

  b_req_tokens_table = torch.arange(0,
                                    batch * max_seq_len,
                                    device='cuda',
                                    dtype=torch.int32).view(batch, max_seq_len)
  b_seq_len = torch.full((batch,),
                         max_seq_len, device='cuda', dtype=torch.int32)
  b_start_loc = torch.tensor([0, max_seq_len, 2 * max_seq_len, 3 * max_seq_len],
                             dtype=torch.int32, device="cuda")

  flash_out = flash_decoding(Q, K_cache, V_cache,
                             scale, b_req_tokens_table, b_seq_len, max_seq_len)
  gloden_out = torch_attention_with_kvcache(Q, K_cache, V_cache,
                                            b_start_loc, b_seq_len)

  if torch.allclose(flash_out, gloden_out, atol=1e-3, rtol=1e-3):
    print("Passed!")
  else:
    diff = (flash_out - gloden_out).abs().max().item()
    print(f"Failed: {diff:.4f}")
  print(f"The maximum difference between torch and triton is \
    {torch.max(torch.abs(gloden_out - flash_out))}")
  if torch.allclose(flash_out, gloden_out, atol=1e-2):
    print(
    "Prefill Stage Test Passed: \
      Triton output matches PyTorch standard implementation.")
  else:
    max_diff = (flash_out - gloden_out).abs().max()
    print(f"Prefill Stage Test Failed: Maximum difference {max_diff}")
  for i in range(0, 32, 1):
    print(f"result = {flash_out[0][i][i]}, ref = {gloden_out[0][i][i]}")

  token_numbers = [64, 128, 256, 512, 1024, max_seq_len]
  plot_performance_comparison(token_numbers,
                              warmup_iterations=10,
                              test_iterations=50)

if __name__ == "__main__":
  FLA_v3_benchmark()
