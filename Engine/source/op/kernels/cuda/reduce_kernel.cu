// baseline: 
// v0带宽利用率: 62.67%
// v0内存吞吐量: 144.33GB/s
// v1: 解决 warp divergent
// v1带宽利用率: 86.46%
// v1内存吞吐量: 199.35GB/s
// v2: 解决 bank conflict
// v2带宽利用率: 89.95%
// v2内存吞吐量: 207.42GB/s

#include <cub/cub.cuh>
#include "tensor/tensor.h"
#include "reduce_kernel.cuh"
#include "base/cuda_config.h"

namespace kernel {
__global__ void reduce_kernel_v0(const float* input, float* output) {
  // 256个thread
  // 256 * 32/8 = 1024Bytes -> 1kb
  // 3080: 单个SM的L1 cache 128kb
  extern __shared__ float smem[];
  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // L3 -> L2
  smem[tid] = input[gid];
  __syncthreads();

  // [1, 2, 3, 4, 5, 6, 7, 8]
  // [1 + 2, 2, 3 + 4, 4, 5 + 6, 6, 7 + 8, 8]
  // [1 + 2 + 3 + 4, 2, 3 + 4, 5 + 6 + 7 + 8, 6, 7 + 8, 8]
  // [1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, 2, 3 + 4, 5 + 6 + 7 + 8, 6, 7 + 8, 8]
  for (uint32_t i = 1; i < blockDim.x; i *= 2) {
    if (tid % (2 * i) == 0) {
      smem[tid] += smem[tid + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

__global__ void reduce_kernel_v1(const float* input, float* output) {
  extern __shared__ float smem[];
  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // L3 -> L2
  smem[tid] = input[gid];
  __syncthreads();

  // 针对v0
  //  在0号warp中, 有一半会进入if, 另一半不会进入 -> divergent
  //  [1, 2, 3, 4, 5, 6, 7, 8]
  //  [1 + 5, 2 + 6, 3 + 7, 4 + 8, 5, 6, 7, 8]
  //  [1 + 5 + 3 + 7, 2 + 6 + 4 + 8, 3 + 7, 4 + 8, 5, 6, 7, 8]
  //  [1 + 5 + 3 + 7 + 2 + 6 + 4 + 8, 2 + 6 + 4 + 8, 3 + 7, 4 + 8, 5, 6, 7, 8]
  // 修改后
  //  前一半的warp会进入if, 后一半的warp不会进入, 直到最后一次, 0号warp会divergent
  for (uint32_t i = 1; i < blockDim.x; i *= 2) {
    uint32_t index = tid * i * 2;
    if (index < blockDim.x) {
      smem[index] += smem[index + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

__global__ void reduce_kernel_v2(const float* input, float* output) {
  extern __shared__ float smem[];
  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // L3 -> L2
  smem[tid] = input[gid];
  __syncthreads();

  // 针对v1
  //  i = 1
  //  0号warp的0号thread: smem[0] += smem[1]
  //  0号warp的16号thread: smem[32] += smem[33]
  //  此时会conflict smem[0] 和 smem[32] 会 conflict

  // eg: thread_num = 128
  // 0号warp不会conflict
  //  0 += 65
  //  31 += 96(也是bank0, 为什么不算conflict)
  //  虽然也是访问bank0, 但是和0号warp不是一个warp,不同warp之间不会在同一个时钟周期, 不同warp之间不会bank conflict
  for (uint32_t i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      smem[tid] += smem[tid + i];
  }
  __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

void reduce_kernel_cu(const tensor::Tensor &input,
                      tensor::Tensor &output,
                      para::reduce_para para,
                      void* stream) {
  CHECK_EQ(input.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  int32_t thread_num = para.thread_num;
  int32_t block_num = para.block_num;

  dim3 grid(block_num);
  dim3 block(thread_num);

  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    reduce_kernel_v2<<<grid, block, thread_num * sizeof(float), stream_>>>(
      input.ptr<float>(), output.ptr<float>());
  } else {
    reduce_kernel_v2<<<grid, block, thread_num * sizeof(float)>>>(
      input.ptr<float>(), output.ptr<float>());
  }
}
} // namespace kernel
