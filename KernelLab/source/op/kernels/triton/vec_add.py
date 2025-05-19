import torch
import triton
import triton.language as tl

# 装饰器: 用于将函数编译为GPU上运行的kernel
# BLOCK_SIZE: tl.constexpr): 编译期常量
@triton.jit
def vector_add_kernel(x_ptr, y_ptr,
                      output_ptr, n_elements,
                      BLOCK_SIZE: tl.constexpr):
  # pid: 全局索引
  pid = tl.program_id(axis=0)

  # 当前block的起始和结束索引
  block_start = pid * BLOCK_SIZE
  offset = tl.arange(0, BLOCK_SIZE)

  # 创建mask防止越界
  #   (SIMT, mask并不是向量的, 是标量, 单指当前这个thread访问的这个数据)
  mask = block_start + offset < n_elements

  x = tl.load(x_ptr + block_start + offset, mask)
  y = tl.load(y_ptr + block_start + offset, mask)

  output = x + y

  tl.store(output_ptr + block_start + offset, output, mask)

def vector_add_host(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
  # checkArgs
  assert x.is_cuda and y.is_cuda
  assert x.shape == y.shape
  assert x.is_contiguous() and y.is_contiguous()

  # compute
  n_elements = x.numel()
  BLOCK_SIZE = 2048 # 可以突破Ampere的1024个thread的限制

  # grid: 必须是一个tuple
  grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)

  vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE)

  return out

if __name__ == "__main__":
  x = torch.rand(512 * 2048, device='cuda')
  y = torch.rand(512 * 2048, device='cuda')
  out = torch.empty_like(x)

  out = vector_add_host(x, y, out)

  assert torch.allclose(out, x + y), "Triton kernel mismatch"
  print("vector_add_kernel successfully")
