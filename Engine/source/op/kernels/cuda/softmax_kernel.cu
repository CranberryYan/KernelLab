#include <cuda_runtime_api.h>
#include "math_utils.cuh"
#include "base/para.h"
#include "softmax_kernel.cuh"

namespace kernel {
// 返回当前这个warp的max
__inline__ __device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__inline__ __device__ float block_reduce_max(float val) {
  __shared__ float shared_max[32]; // Ampere最多分配1024个thread -> 32个warp
  uint32_t lane_id = threadIdx.x % 32;
  uint32_t wid = threadIdx.x / 32;

  for (int i = 0; i < 32; ++i) {
    shared_max[i] = -INFINITY;
  }
  __syncthreads();

  val = warp_reduce_max(val); // val: 当前这个wid中的最大值
  if (lane_id == 0) {
    shared_max[wid] = val;
  }
  __syncthreads();

  // 赋值, 把shared_max的值赋给前32个thread(第一个warp), 让其规约
  // shared_max: 
  val = (threadIdx.x < 32) ? shared_max[threadIdx.x] : -INFINITY;

  if (wid == 0) {
    val = warp_reduce_max(val); // 拿到当前block的max
  }

  return (threadIdx.x == 0) ? val : -INFINITY;
}

// 返回当前这个warp的所有值的sum
__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
  __shared__ float shared_sum[32];
  uint32_t lane_id = threadIdx.x % 32;
  uint32_t wid = threadIdx.x / 32;

  for (int i = 0; i < 32; ++i) {
    shared_sum[i] = 0.0f;
  }
  __syncthreads();

  val = warp_reduce_sum(val); // val: 当前这个wid中的所有元素的sum
  if (lane_id == 0) {
    shared_sum[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < 32) ? shared_sum[threadIdx.x] : 0;

  val = warp_reduce_sum(val);

  return val;
}

__global__ void naive_softmax_kernel_v0(const float* input, float* output,
                                        const uint32_t rows,
                                        const uint32_t cols) {
  uint32_t bid = blockIdx.x;
  uint32_t tid = threadIdx.x;

  // Naive版本, 仅需要求出sum
  __shared__ float sum;
  extern __shared__ float shared_cols[];
  for (int r = bid; r < rows; r += gridDim.x) {
    if (tid == 0) {
      sum = 0;
    }
    for (int c = tid; c < cols; c += blockDim.x) {
      shared_cols[c] = expf(input[r * cols + c]);
    }
    __syncthreads();

    if (tid == 0) {
      for (int c = 0; c < cols; ++c) {
        sum += shared_cols[c];
      }
    }
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
      output[r * cols + c] = shared_cols[c] / sum;
    }
    __syncthreads();
  }
}

__global__ void safe_softmax_kernel_v0(const float* input, float* output,
                                       const uint32_t rows,
                                       const uint32_t cols) {
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;

  __shared__ float sum;
  __shared__ float max_val;
  extern __shared__ float shared_cols[];
  for (int r = bid; r < rows; r += gridDim.x) {
    if (tid == 0) {
      sum = 0;
      max_val = -INFINITY;
    }
    for (int c = tid; c < cols; c += blockDim.x) {
      shared_cols[c] = input[r * cols + c];
    }
    __syncthreads();

    for (int c = 0; c < cols; ++c) {
      max_val = max(max_val, shared_cols[c]);
    }
    __syncthreads();

    for (int c = 0; c < cols; ++c) {
      sum += expf(shared_cols[c] - max_val);
    }
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
      output[r * cols + c] = expf(shared_cols[c] - max_val) / sum;
    }
  }
}

__global__ void safe_softmax_kernel_v1(const float* input, float* output,
                                       const uint32_t rows,
                                       const uint32_t cols) {
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;

  __shared__ float sum_block;
  __shared__ float max_block;
  extern __shared__ float shared_cols[];

  uint32_t cols_4 = cols / 4;
  for (int r = bid; r < rows; r += gridDim.x) {
    for (int c = tid; c < cols_4; c += blockDim.x) {
      shared_cols[c * 4 + 0] = input[r * cols + c * 4 + 0];
      shared_cols[c * 4 + 1] = input[r * cols + c * 4 + 1];
      shared_cols[c * 4 + 2] = input[r * cols + c * 4 + 2];
      shared_cols[c * 4 + 3] = input[r * cols + c * 4 + 3];
    }
    for (int c = cols_4 * 4 + tid; c < cols; c += blockDim.x) {
      shared_cols[c] = input[r * cols + c];
    }
    __syncthreads();

    float max_tmp = block_reduce_max(shared_cols[tid]);
    if (tid == 0) {
      max_block = max_tmp;
    }
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
      shared_cols[c] = expf(shared_cols[c] - max_block);
    }
    __syncthreads();

    float sum_tmp = block_reduce_sum(shared_cols[tid]);
    if (tid == 0) {
      sum_block = sum_tmp;
    }
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
      output[r * cols + c] = shared_cols[c] / sum_block;
    }
  }
}

__global__ void online_softmax_kernel_v0(const float* input, float* output,
                                         const uint32_t rows,
                                         const uint32_t cols) {
  // 一个thread负责一行
  //  FlashAttention中, 行比较大, 列比较小
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t r = gid;
  if (r >= rows) {
    return;
  }

  float sum = 0;
  float max_tmp = -INFINITY;
  float pre_max_tmp = 0;
  for (int c = 0; c < cols; ++c) {
    uint32_t input_offset = r * cols + c;
    max_tmp = max(max_tmp, input[input_offset]);
    sum =
      sum * expf(pre_max_tmp - max_tmp) +
        expf(input[input_offset] - max_tmp);
    pre_max_tmp = max_tmp;
  }

  for (int c = 0; c < cols; ++c) {
    uint32_t input_offset = r * cols + c;
    output[input_offset] = expf(input[input_offset] - max_tmp) / sum;
  }
}

__global__ void online_softmax_kernel_v1(const float* input, float* output,
                                         const uint32_t rows,
                                         const uint32_t cols) {
  // 一个blokc负责n行, 最多1024个block, stride: gridDim.x
  //  一个block中的thread负责n个元素, 最多128个thread, stride: blockDim.x
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  if (bid >= rows || tid >= cols) {
    return;
  }

  extern __shared__ float shared_cols[];
  __shared__ float sum;
  __shared__ float max_tmp;
  __shared__ float pre_max_tmp;
  for (int b = bid; b < rows; b += gridDim.x) {
    if (tid == 0) {
      sum = 0;
      max_tmp = -INFINITY;
      pre_max_tmp = 0;
    }
    for (int c = tid; c < cols; c += blockDim.x) {
      shared_cols[c] = input[b * cols + c];
    }
    __syncthreads();
    if (tid == 0) {
      for (int c = 0; c < cols; ++c) {
        max_tmp = max(max_tmp, shared_cols[c]);
        sum =
          sum * expf(pre_max_tmp - max_tmp) +
            expf(shared_cols[c] - max_tmp);
        pre_max_tmp = max_tmp;
      }
    }
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
      output[b * cols + c] = expf(shared_cols[c] - max_tmp) / sum;
    }
    __syncthreads();
  }
}

void softmax_kernel_cu(const tensor::Tensor& input,
                        tensor::Tensor& output,
                        para::softmax_para para,
                        void* stream) {
  uint32_t rows = para.input_rows;
  uint32_t cols = para.input_cols;

  uint32_t thread_num = cols < 512 ? cols : 512;
  uint32_t block_num = rows < 1024 ? rows : 1024;

  dim3 block(thread_num);
  dim3 grid(block_num);

  uint32_t smem_size = math_cu::AlignUp<uint32_t>(cols * sizeof(float), 512);

  if (para.op_type == para::SoftmaxOpType::Naive) {
    if (stream) {
      cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
      naive_softmax_kernel_v0<<<grid, block, smem_size, stream_>>>(
        input.ptr<float>(), output.ptr<float>(), rows, cols);
    } else {
      naive_softmax_kernel_v0<<<grid, block, smem_size>>>(
        input.ptr<float>(), output.ptr<float>(), rows, cols);
    }
  } else if (para.op_type == para::SoftmaxOpType::Safe) {
    if (stream) {
      cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
      safe_softmax_kernel_v1<<<grid, block, smem_size, stream_>>>(
        input.ptr<float>(), output.ptr<float>(), rows, cols);
    } else {
      safe_softmax_kernel_v1<<<grid, block, smem_size>>>(
        input.ptr<float>(), output.ptr<float>(), rows, cols);
    }
  } else if (para.op_type == para::SoftmaxOpType::Online) {
    if (stream) {
      cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
      online_softmax_kernel_v1<<<grid, block, smem_size, stream_>>>(
        input.ptr<float>(), output.ptr<float>(), rows, cols);
    } else {
      online_softmax_kernel_v1<<<grid, block, smem_size>>>(
        input.ptr<float>(), output.ptr<float>(), rows, cols);
    }
  } else {
    printf("ERROR: Unknown SoftmaxOpType\n");
  }
}
} // namespace kernel
