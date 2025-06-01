import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton import Config
import numpy as np

# Llama的Gate Linear
# out = W2(GELU(W1a(x)) ⊙ W1b(x))
class MLP(nn.Module):
  def __init__(self, dim: int, hidden_dim: int, mutiple_of: int):
    super().__init__()
    # hidden_dim:
    # Llama中采用Gate Linear, 一个做act, 另一个做gating, 点乘下投影回dim
    # 为了减小计算开销, hidden_dim -> 2/3 hidden_dim
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = mutiple_of * ((hidden_dim + mutiple_of - 1) // mutiple_of)

    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x):
    w1x = F.silu(self.w1(x)) # act
    w3x = self.w3(x)         # gating
    out = self.w2(w1x * w3x)

    return out

# silu
class sigmoid(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 1 / (1 + torch.exp(-x))

class silu(nn.Module):
  def __init__(self):
    super().__init__()
    self.sigmoid = sigmoid()

  def forward(self, x):
    return x * self.sigmoid(x)

@triton.jit
def silu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offset = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offset < n_elements

  x = tl.load(x_ptr + offset, mask, other=0.0)

  x_fp32 = x.to(tl.float32)
  sigmoid_x =  1 / (1 + tl.exp(-x_fp32))
  y = x_fp32 * sigmoid_x
  y = y.to(tl.float16)

  tl.store(output_ptr + offset, y, mask)

def silu_host(x: torch.Tensor):
  output = torch.empty_like(x)

  n_elements = x.numel()
  BLOCK_SIZE = 1024
  grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']), 1, 1)
  silu_kernel[grid](x, output, n_elements, BLOCK_SIZE)

  return output

# mul
@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offset < n_elements

  x = tl.load(x_ptr + offset, mask, 0.0)
  x_fp32 = x.to(tl.float32)
  y = tl.load(y_ptr + offset, mask, 0.0)
  y_fp32 = y.to(tl.float32)

  acc_fp32 = x_fp32 * y_fp32
  acc = acc_fp32.to(tl.float16)

  tl.store(output_ptr + offset, acc, mask)

def mul_host(x: torch.Tensor, y: torch.Tensor):
  output = torch.empty_like(x)

  n_elements = x.numel()
  BLOCK_SIZE = 1024
  grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']), 1, 1)
  mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)

  return output

# mlp
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

@triton.autotune(configs=get_configs(), key=["M", "N", "K"])
@triton.jit
def gemm_kernel(x_ptr, w_ptr, y_ptr, M, N, K,
               BLOCK_SIZE_M: tl.constexpr,
               BLOCK_SIZE_N: tl.constexpr,
               BLOCK_SIZE_K: tl.constexpr):
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)

  m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  k_offset = tl.arange(0, BLOCK_SIZE_K)
  y_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

  for k in range(0, K, BLOCK_SIZE_K):
    # slice: 
    # x: [BLOCK_SIZE_M, BLOCK_SIZE_K]
    #   [BLOCK_SIZE_M, 1] + [1, BLOCK_SIZE_K] -> 广播
    #     -> [BLOCK_SIZE_M, BLOCK_SIZE_K]
    # w: [BLOCK_SIZE_N, BLOCK_SIZE_K](要在load时, 进行逻辑上的转置)
    #   [BLOCK_SIZE_K, BLOCK_SIZE_N]
    # y: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    # x_ptr + (m_offset[:, None] * K + k_offset[None, :] + k)
    #   类似于燧原的全量配置
    #   如果改成增量, 那么步长为BLOCK_SIZE_K
    x = x_ptr + (m_offset[:, None] * K + k_offset[None, :] + k)
    mask_x = (m_offset[:, None] < M) & (k_offset[None, :] + k < K)
    x_tile = tl.load(x, mask_x, other=0.0)

    w = w_ptr + (n_offset[None, :] * K + k_offset[:, None] + k)
    mask_w = (n_offset[None, :] < N) & (k_offset[:, None] + k < K)
    w_tile = tl.load(w, mask_w, other=0.0)

    y_tile += tl.dot(x_tile, w_tile)

  tl.store(y_ptr + m_offset[:, None] * N + n_offset[None, :],
           y_tile.to(tl.float16),
           ((m_offset[:, None] < M) & (n_offset[None, :] < N)))

def gemm_host(x: torch.Tensor, w: torch.Tensor, M, N, K):
  y = torch.empty((M, N), dtype=torch.float16, device='cuda')

  grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),
                       triton.cdiv(N, META['BLOCK_SIZE_N']),
                       1)
  gemm_kernel[grid](x, w, y, M, N, K)

  return y

# UT
def silu_test(x: torch.Tensor):
  res = torch.empty_like(x)
  ref = torch.empty_like(x)

  ref = F.silu(x).cpu().numpy()
  res = silu_host(x).cpu().numpy()

  mask = ~np.isclose(ref, res, atol=1e-5, rtol=1e-5)
  mismatch_indices = np.argwhere(mask)

  print(" ================== enter mul UT ==================")
  print("Mismatched elements:", np.sum(mask))
  if np.sum(mask):
    for idx in mismatch_indices[:10]:
      idx_tuple = tuple(idx)
      val_res = res[idx_tuple]
      val_ref = ref[idx_tuple]
      diff = abs(val_res - val_ref)
      print(f"Mismatch at {idx_tuple}: \
        result = {val_res}, ref = {val_ref}, diff = {diff}")
  else:
    for i in range(0, 10, 1):
      print(f"result = {res[0][i]}, ref = {ref[0][i]}")

def mul_test(x: torch.Tensor, y: torch.Tensor):
  res = torch.empty_like(x)
  ref = torch.empty_like(x)

  ref = (x * y).cpu().numpy()
  res = mul_host(x, y).cpu().numpy()

  mask = ~np.isclose(ref, res, atol=1e-5, rtol=1e-5)
  mismatch_indices = np.argwhere(mask)

  print(" ================== enter mul UT ==================")
  print("Mismatched elements:", np.sum(mask))
  if np.sum(mask):
    for idx in mismatch_indices[:10]:
      idx_tuple = tuple(idx)
      val_res = res[idx_tuple]
      val_ref = ref[idx_tuple]
      diff = abs(val_res - val_ref)
      print(f"Mismatch at {idx_tuple}: \
        result = {val_res}, ref = {val_ref}, diff = {diff}")
  else:
    for i in range(0, 10, 1):
      print(f"result = {res[0][i]}, ref = {ref[0][i]}")

def gemm_test(x: torch.Tensor, y: torch.Tensor, M, N, K):
  res = torch.empty((M, N), dtype=torch.float16, device='cuda')
  ref = torch.empty((M, N), dtype=torch.float16, device='cuda')

  ref = (x @ y.T).cpu().numpy()
  res = gemm_host(x, y, M, N, K).cpu().numpy()

  mask = ~np.isclose(ref, res, atol=1e-5, rtol=1e-5)
  mismatch_indices = np.argwhere(mask)

  print(" ================== enter mul UT ==================")
  print("Mismatched elements:", np.sum(mask))
  if np.sum(mask):
    for idx in mismatch_indices[:10]:
      idx_tuple = tuple(idx)
      val_res = res[idx_tuple]
      val_ref = ref[idx_tuple]
      diff = abs(val_res - val_ref)
      print(f"Mismatch at {idx_tuple}: \
        result = {val_res}, ref = {val_ref}, diff = {diff}")
  else:
    for i in range(0, 10, 1):
      print(f"result = {res[0][i]}, ref = {ref[0][i]}")

if __name__ == "__main__":
  M = 2048
  N = 2048
  K = 128
  x = torch.rand((M, K), dtype=torch.float16, device='cuda')
  w = torch.rand((N, K), dtype=torch.float16, device='cuda')

  silu_test(x)
  mul_test(x, w)
  gemm_test(x, w, M, N, K)
