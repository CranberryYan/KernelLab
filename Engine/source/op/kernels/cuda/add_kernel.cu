#include "add_kernel.cuh"
#include <cuda_runtime_api.h>

namespace kernel {
__global__ void add_kernel(int32_t size,
                           const float* input1, const float* input2,
                           float* output) {
  int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid > size) { return; }
  output[gid] = input1[gid] + input2[gid];
}

void add_kernel_cu(const tensor::Tensor& input1,
                   const tensor::Tensor& input2,
                   tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  CHECK_EQ(input1.size(), input2.size());
  CHECK_EQ(input1.size(), output.size());

  int32_t size = static_cast<int32_t>(input1.size());
  int32_t thread_num = 1024;
  int32_t block_num = (size + thread_num - 1) / thread_num;

  dim3 grid(block_num);
  dim3 block(thread_num);
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel<<<grid, block, 0, stream_>>>(
      size, input1.ptr<float>(), input2.ptr<float>(), output.ptr<float>());
  } else {
    add_kernel<<<grid, block>>>(
      size, input1.ptr<float>(), input2.ptr<float>(), output.ptr<float>());
  }
}
} // namespace kernel