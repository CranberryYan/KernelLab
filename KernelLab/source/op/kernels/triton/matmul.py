import torch
import triton
import numpy as np
import triton.language as tl

def matrix_multiply_baseline(lhs, rhs):
  # lhs rhs 均为二维tensor
  # lhs: [M, K]   rhs: [K, N]
  # 规约轴使用线性规约(注意规约顺序不同导致最后结果不同)
  lhs_shape = lhs.shape
  rhs_shape = rhs.shape
  M = lhs_shape[0]
  N = rhs_shape[1]
  K = lhs_shape[1]
  C = np.zeros((M, N))
  for m in range(M):
    for n in range(N):
      for k in range(K):
        C[m][n] += lhs[m][k] * rhs[k][n]

  return C

def matrix_multiply_block(lhs, rhs, BLOCK_SIZE):
  # lhs rhs 均为二维tensor
  # lhs: [M, K]   rhs: [K, N]
  # 规约轴使用线性规约(注意规约顺序不同导致最后结果不同)
  lhs_shape = lhs.shape
  rhs_shape = rhs.shape
  M = lhs_shape[0]
  N = rhs_shape[1]
  K = lhs_shape[1]
  C = np.zeros((M, N), dtype=lhs.dtype)

  for m in range(0, M, BLOCK_SIZE):
    for n in range(0, N, BLOCK_SIZE):
      tile = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=lhs.dtype)
      for k in range(0, K, BLOCK_SIZE):
        for m_ in range(BLOCK_SIZE):
          for n_ in range(BLOCK_SIZE):
            for k_ in range(BLOCK_SIZE):
              tile[m_][n_] += lhs[m + m_][k + k_] * rhs[k + k_][n + n_]
      for m_ in range(BLOCK_SIZE):
        for n_ in range(BLOCK_SIZE):
          C[m + m_][n + n_] += tile[m_][n_]

  return C

@triton.jit
def linear_kernel(x_ptr, w_ptr, out_ptr, M, N, K,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  # 添加维度
  # offset_m: [BLOCK_SIZE_M, 1]
  #  取哪几行
  # offset_n: [1, BLOCK_SIZE_N]
  #  取哪几列
  offset_m = tl.arange(0, BLOCK_SIZE_M)[:, None] + pid_m * BLOCK_SIZE_M
  offset_n = tl.arange(0, BLOCK_SIZE_N)[None, :] + pid_n * BLOCK_SIZE_N

  tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
  for k in range(0, K, BLOCK_SIZE_K):
    # x_k: [1, BLOCK_SIZE_K]
    x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k

    # offset_m * K
    #  对应的行偏移
    # x_k
    #  对应的列偏移, 每次循环增加BLOCK_SIZE_K
    # offset_m * K: [BLOCK_SIZE_M, 1]   x_k: [1, BLOCK_SIZE_K]
    # offset_m * K + x_k -> 广播 -> [BLOCK_SIZE_M, BLOCK_SIZE_K]
    # x: load一块地址, 类似于燧原的slice出一块
    # shape为[BLOCK_SIZE_M, BLOCK_SIZE_K]的tile
    x = tl.load(x_ptr + offset_m * K + x_k,
                mask=((offset_m < M) & (x_k < K)), other=0.0)

    # w_k * N
    #  对应的列偏移
    # offset_n:
    #  对应的行偏移
    # w_k * N: [BLOCK_SIZE_K, 1]
    # offset_n: [1, BLOCK_SIZE_N]
    # w: [BLOCK_SIZE_K, BLOCK_SIZE_N]
    w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
    w = tl.load(w_ptr + w_k * N + offset_n,
                mask=(offset_n < N) & (w_k < K), other=0.0)

    tile = tl.dot(x, w, acc=tile)

  # out_offset: 广播 —> [BLOCK_SIZE_M, BLOCK_SIZE_N]
  out_offset = offset_m * N + offset_n
  tl.store(out_ptr + out_offset, tile, ((offset_m < M) & (offset_n < N)))

def linear_kernel_host(lhs, rhs):
  lhs_ = lhs.view((-1, lhs.shape[-1]))
  M, K = lhs_.shape
  K, N = rhs.shape
  output = torch.empty((M, N), device=lhs.device, dtype=lhs.dtype)

  BLOCK_SIZE_M = 64
  BLOCK_SIZE_N = 64
  BLOCK_SIZE_K = 32

  grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
  linear_kernel[grid](lhs, rhs, output, M, N, K,
                      BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

  output = output.view((lhs.shape[0], lhs.shape[1], N))

  return output

# Triton
# Triton以Block作为基本单位, 我们配置一个二维网格
# 第一维: M轴的block数量, 第二维: N轴的block数量
if __name__ == '__main__':
  batch_size = 64
  seq_len = 128
  hiddem_dim = 1280
  output_dim = 2560

  input = torch.rand((batch_size, seq_len, hiddem_dim),
                      device='cuda', dtype=torch.float32)

  weight = torch.rand((hiddem_dim, output_dim),
                       device='cuda', dtype=torch.float32)

  output_res = linear_kernel_host(input, weight)
  output_ref = (input @ weight).cpu().numpy()
  output_res_host = output_res.cpu().numpy()

  mask = ~np.isclose(output_res_host, output_ref, atol=1e-2, rtol=1e-2)

  print("Mismatched elements:", np.sum(mask))

  mismatch_indices = np.argwhere(mask)

  for idx in mismatch_indices[:10]:
    idx_tuple = tuple(idx)
    val_res = output_res_host[idx_tuple]
    val_ref = output_ref[idx_tuple]
    diff = abs(val_res - val_ref)
    print(f"Mismatch at {idx_tuple}: \
      result = {val_res}, ref = {val_ref}, diff = {diff}")
