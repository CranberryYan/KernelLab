import torch
import triton
import triton.language as tl
from triton import Config
import numpy as np

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

def get_configs():
  return [
    Config(
      {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32
      },
      num_warps=4,
      num_stages=2
    ),
    Config(
      {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32
      },
      num_warps=8,
      num_stages=3
    ),
    Config(
      {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
      },
      num_warps=4,
      num_stages=4
    )
  ]

@triton.jit
def relu(x):
  # where: 类似于vsel
  return tl.where(x > 0, x, 0)

@triton.jit
def leaky_relu(x):
  # where: 类似于vsel
  return tl.where(x > 0, x, 0.01 * x)

@triton.autotune(configs=get_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel_autotune(x_ptr, w_ptr, out_ptr,
                           M, N, K,
                           BLOCK_SIZE_M: tl.constexpr,
                           BLOCK_SIZE_N: tl.constexpr,
                           BLOCK_SIZE_K: tl.constexpr,
                           ACTIVATION: tl.constexpr = "relu"):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  m_offset = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
  n_offset = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
  k_offset = tl.arange(0, BLOCK_SIZE_K)

  x_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float16)
  w_tile = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16)
  out_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

  for k in range(0, K, BLOCK_SIZE_K):
    # slice
    # x: 
    # [BLOCK_SIZE_M, 1] + [1, BLOCK_SIZE_K] -> [BLOCK_SIZE_M, BLOCK_SIZE_K]
    # 行并行, 列随着循环移动
    x = x_ptr + (m_offset[:, None] * K + k_offset[None, :] + k)
    mask_x = (m_offset[:, None] < M) & (k_offset[None, :] + k < K)
    x_tile = tl.load(x, mask_x, other=0.0)

    # w:
    # [BLOCK_SIZE_K, 1] + [1, BLOCK_SIZE_N] -> [BLOCK_SIZE_K, BLOCK_SIZE_N]
    # 列并行, 行随着循环移动
    w = w_ptr + ((k_offset[:, None] + k) * N + n_offset[None, :])
    mask_w = (k_offset[:, None] + k < K) & (n_offset[None, :] < N)
    w_tile = tl.load(w, mask_w, other=0.0)

    out_tile = tl.dot(x_tile, w_tile, acc=out_tile)

  if ACTIVATION == "relu":
    out_tile = relu(out_tile)
  elif ACTIVATION == "leaky_relu":
    out_tile = leaky_relu(out_tile)

  # out:
  # [BLOCK_SIZE_M, 1] + [1, BLOCK_SIZE_N] -> [BLOCK_SIZE_M, BLOCK_SIZE_N]
  out = out_ptr + m_offset[:, None] * N + n_offset[None, :]
  mask_out = (m_offset[:, None] < M) & (n_offset[None, :] < N)
  tl.store(out, out_tile.to(tl.float16), mask_out)

def linear_kernel_host(lhs, rhs, activation="relu"):
  lhs_ = lhs.view((-1, lhs.shape[-1]))
  M, K = lhs_.shape
  K, N = rhs.shape
  output = torch.empty((M, N), device=lhs.device, dtype=lhs.dtype)

  grid = lambda META: (
    triton.cdiv(M, META["BLOCK_SIZE_M"]),
    triton.cdiv(N, META["BLOCK_SIZE_N"]),
    1)
  matmul_kernel_autotune[grid](lhs, rhs, output, M, N, K, ACTIVATION=activation)

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
                      device='cuda', dtype=torch.float16)

  weight = torch.rand((hiddem_dim, output_dim),
                       device='cuda', dtype=torch.float16)

  output_res = linear_kernel_host(input, weight, "leaky_relu")
  output_ref = (input @ weight).cpu().numpy()
  output_res_host = output_res.cpu().numpy()

  mask = ~np.isclose(output_res_host, output_ref, atol=1e-2, rtol=1e-2)

  print("Mismatched elements:", np.sum(mask))

  mismatch_indices = np.argwhere(mask)

  if np.sum(mask):
    for idx in mismatch_indices[:10]:
      idx_tuple = tuple(idx)
      val_res = output_res_host[idx_tuple]
      val_ref = output_ref[idx_tuple]
      diff = abs(val_res - val_ref)
      print(f"Mismatch at {idx_tuple}: \
        result = {val_res}, ref = {val_ref}, diff = {diff}")
  else:
    for i in range(0, 10, 1):
      print(f"result = {output_res_host[0][i]}, ref = {output_ref[0][i]}")
