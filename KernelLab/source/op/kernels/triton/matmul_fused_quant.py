# TODO: gemm和dequant的融合部分还有bug

# Cq = Quantize_fp16(Dequantize_fp16(Aq, Sa) × Dequantize_fp16(Wq, Sw))
# Aq和Wq是量化后的输入和权重, 均为fp16
# Sa和Sw是量化参数, 均为fp32
# 先反量化将Aq和Wq: fp16 -> fp32 -> 进行计算 -> 量化 -> fp32 -> fp16

# FP8是一种用于深度学习的低精度数值格式, 主要有两种变体:
# E4M3: 由 1 位符号位、4 位指数位和 3 位尾数位组成。
# E5M2: 由 1 位符号位、5 位指数位和 2 位尾数位组成。
import os
import torch
import triton
import triton.language as tl
from triton import Config
from typing import Tuple

os.environ['TRITON_LOG_LEVEL'] = 'debug'

FP8_E4M3_MAX = 448.0    # FP8 E4M3 格式的最大可表示值
FP8_E5M2_MAX = 57344.0  # FP8 E5M2 格式的最大可表示值

# 切分策略
#  行并行, 每个block负责一行
#  BLOCK_SIZE: self_cols
# s_ptr: 分组量化, 每BLOCK_SIZE使用一个scale
@triton.jit
def quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  offset_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  offset_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

  tile_n = tl.cdiv(N, BLOCK_SIZE)

  # 计算二维偏移
  offsets = offset_m[:, None] * N + offset_n[None, :]
  mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)

  x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)

  s = tl.max(tl.abs(x))
  y = x / s
  y = y.to(y_ptr.dtype.element_ty)

  tl.store(y_ptr + offsets, y, mask)
  tl.store(s_ptr + pid_m * tile_n + pid_n, s)

def quant_host(x: torch.Tensor, block_size: int = 128) ->\
  Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'\
      Last dimension size must be divisible by block_size\
        (block_size={block_size})'

    M, N = x.size()
    # tips: 如果使用torch.empty, tensor默认在CPU上
    y = torch.empty_like(x, dtype=torch.float16)

    # *: 元组拆开作为函数参数
    #  dims = (32, 64)   foo(*dims) == foo(32, 64)
    # x: [B, H, W]
    # s: [B, H, W // block_size]
    s = x.new_empty(*x.size()[:-1],
                    x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']),
                        triton.cdiv(N, META['BLOCK_SIZE']),
                        1)

    quant_kernel[grid](x, y, s, M, N, BLOCK_SIZE=block_size)

    return y, s

@triton.jit
def dequant_kernel(x_ptr, s_ptr, y_ptr, M, N,
                   BLOCK_SIZE: tl.constexpr):
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)

  offset_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  offset_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

  # s: [M, N // BLOCK_SIZE]
  tile_n = tl.cdiv(N, BLOCK_SIZE) # 每行使用的scale数(s的列数)

  # deslice: offset: [BLOCK_SIZE, BLOCK_SIZE]
  offset = offset_m[:, None] * N + offset_n[None, :]
  mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)

  x = tl.load(x_ptr + offset, mask, other=0.0).to(tl.float32)
  s = tl.load(s_ptr + pid_m * tile_n + pid_n).to(tl.float32)

  y = x * s

  tl.store(y_ptr + offset, y, mask)

def dequant_host(x: torch.Tensor, s: torch.Tensor,
                 block_size: int = 128) -> torch.Tensor:
  assert x.is_contiguous() and s.is_contiguous(), \
    'Input tensors must be contiguous'
  assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'

  M, N = x.size()
  y = torch.empty_like(x)

  grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']),
                       triton.cdiv(N, META['BLOCK_SIZE']),
                       1)

  dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)

  return y

fp16_gemm_configs = [
  Config({'BLOCK_SIZE_M': block_m,
          'BLOCK_SIZE_N': block_n,
          'BLOCK_SIZE_K': 128},
         num_stages=num_stages, num_warps=8)
  for block_m in [16, 32, 64]
  for block_n in [32, 64, 128]
  for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp16_gemm_configs, key=['N', 'K'])
@triton.jit
def fp16_gemm_kernel(x_ptr, w_ptr, y_ptr,
                     x_s_ptr, w_s_ptr,
                     M,
                     N: tl.constexpr,
                     K: tl.constexpr,
                     BLOCK_SIZE_M: tl.constexpr,
                     BLOCK_SIZE_N: tl.constexpr,
                     BLOCK_SIZE_K: tl.constexpr):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  tile_k_num = tl.cdiv(K, BLOCK_SIZE_K)
  offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  offset_k = tl.arange(0, BLOCK_SIZE_K)

  # delice:
  # x: [BLOCK_SIZE_M, BLOCK_SIZE_K]

  # 逻辑上进行转置
  # w_ptr: [N, K]
  # w_ptrs: [BLOCK_SIZE_K, BLOCK_SIZE_N]
  # 每BLOCK_SIZE_K个元素用一个scale
  # x_ptrs: 行
  # w_ptrs: 列
  x_ptrs = x_ptr + offset_m[:, None] * K + offset_k[None, :]
  w_ptrs = w_ptr + offset_n[None, :] * K + offset_k[:, None]

  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

  for k_block in range(tile_k_num):
    k_offset = k_block * BLOCK_SIZE_K

    mask_x = (offset_k[None, :] + k_offset) < K
    mask_w = (offset_k[:, None] + k_offset) < K

    x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    x_s = tl.load(x_s_ptr + pid_m * tile_k_num + k_block)
    w_s = tl.load(w_s_ptr + pid_n * tile_k_num + k_block)

    accumulator += tl.dot(x, w) * x_s[:, None] * w_s[None, :]

    # 类似于燧原的增量配置(更好, 开销更小)
    x_ptrs += BLOCK_SIZE_K
    w_ptrs += BLOCK_SIZE_K

  out = accumulator.to(y_ptr.dtype.element_ty)

  out_ptrs = y_ptr + offset_m[:, None] * N + offset_n[None, :]
  mask_out = (offset_m[:, None] < M) & (offset_n[None, :] < N)
  tl.store(out_ptrs, out, mask_out)

def fp16_gemm_host(x: torch.Tensor, x_s: torch.Tensor,
                  w: torch.Tensor, w_s: torch.Tensor):
  assert x.is_contiguous() and w.is_contiguous(),\
    'Input tensors must be contiguous'
  assert x_s.is_contiguous() and w_s.is_contiguous(),\
    'Scaling factor tensors must be contiguous'

  M = x.size(0)
  N = w.size(0)
  K = w.size(1)
  y = x.new_empty(*x.size()[:-1], N, dtype=torch.get_default_dtype())

  grid = lambda META:\
    (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

  fp16_gemm_kernel[grid](x, w, y, x_s, w_s, M, N, K)

  return y

def test_quant_dequant():
  torch.manual_seed(0)
  block_size = 128

  x = torch.randn((128, 256), dtype=torch.float32, device='cuda')

  y, s = quant_host(x, block_size=block_size)

  y_dequant = dequant_host(y, s, block_size=block_size)

  max_error = (x - y_dequant).abs().max().item()
  mse = ((x - y_dequant) ** 2).mean().item()

  print(f"Max absolute error after quant-dequant: {max_error:.6f}")
  print(f"Mean squared error after quant-dequant: {mse:.8f}")

  assert max_error < 0.1, "Max error too large"
  assert mse < 0.01, "MSE too large"

  print("Quantization-Dequantization test passed!")

def test_fp16_gemm():
  torch.manual_seed(0)
  block_size = 128

  M, K, N = 128, 256, 64
  x = torch.randn((M, K), dtype=torch.float32, device='cuda')
  w = torch.randn((N, K), dtype=torch.float32, device='cuda')

  x_q, x_s = quant_host(x, block_size=block_size)
  w_q, w_s = quant_host(w, block_size=block_size)

  one_x = torch.ones_like(x_s)
  one_w = torch.ones_like(w_s)

  x_dequant = dequant_host(x_q, x_s, block_size=block_size)
  w_dequant = dequant_host(w_q, w_s, block_size=block_size)

  y_fp16 = fp16_gemm_host(x_dequant, one_x, w_dequant, one_w)

  y_ref = x @ w.T

  max_error = (y_ref - y_fp16).abs().max().item()
  mse = ((y_ref - y_fp16) ** 2).mean().item()

  print(f"FP8 GEMM max absolute error: {max_error:.6f}")
  print(f"FP8 GEMM mean squared error: {mse:.8f}")

  assert max_error < 0.1, "FP8 GEMM max error too large"
  assert mse < 0.01, "FP8 GEMM MSE too large"

  print("FP8 GEMM test passed!")

if __name__ == "__main__":
  test_quant_dequant()
  test_fp16_gemm()
