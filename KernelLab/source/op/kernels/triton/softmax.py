import torch
import triton
import triton.language as tl

def assert_allclose_debug(ref, res, atol=1e-5, rtol=1e-5, max_print=10):
  mask = ~torch.isclose(ref, res, atol=atol, rtol=rtol)
  if mask.any():
    idxs = mask.nonzero(as_tuple=False)
    ref_vals = ref[mask]
    res_vals = res[mask]
    print(f"Found {idxs.size(0)} mismatches; showing up to {max_print}:")
    for idx, rv, sv in zip(idxs[:max_print], ref_vals[:max_print], res_vals[:max_print]):
      coords = tuple(idx.tolist())
      print(f"  idx={coords}: ref={rv:.6f}, res={sv:.6f}, diff={abs(rv-sv):.6e}")
    raise AssertionError("Triton kernel mismatch")
  else:
    print("All values match within tolerance!")

# SoftMax(x) = e^(x-max) / sum(e^(x-max))
def softmax_torch(x):
  x_exp = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
  return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

@triton.jit
def softmax_triton(input_ptr, output_ptr,
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr):
  # 切分策略:
  #   每个block计算2行
  row_per_block = 2
  row_start = tl.program_id(axis=0) * row_per_block
  if row_start > n_rows:
    return

  for row_idx in range(row_start, row_start + row_per_block, 1):
    input_row_start_ptr = input_ptr + row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptr_ = input_row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    
    # 因为tl中的max底层使用规约算法, 因此BLOCK_SIZE必须是2的整数次幂
    # 导致分配的BLOCK_SIZE可能会远大于n_cols, 所以使用mask来控制
    # mask为false的位置, 会选择other中的值, 为了接下来求max, other中为-Inf
    row = tl.load(input_ptr_, mask, other=-float('inf'))

    row_ = row - tl.max(row, axis=0)
    numerator = tl.exp(row_)
    denominator = tl.sum(numerator, axis=0)
    res = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * n_cols
    output_ptr_ = output_row_start_ptr + col_offsets
    tl.store(output_ptr_, res, mask)

if __name__ == "__main__":
  input = torch.randn(1000, 513, device='cuda')
  res = torch.empty_like(input)
  ref = softmax_torch(input)

  n_rows, n_cols = input.shape
  BLOCK_SIZE = triton.next_power_of_2(n_cols) # 不小于n_cols的2的整数次幂

  BLOCK_NUMS = triton.cdiv(n_rows, 2)
  grid = (BLOCK_NUMS, 1, 1)

  softmax_triton[grid](input, res, n_rows, n_cols, BLOCK_SIZE)

  assert_allclose_debug(ref, res)
