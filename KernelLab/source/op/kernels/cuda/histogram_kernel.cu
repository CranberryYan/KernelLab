#include <cub/cub.cuh>
#include "math_utils.cuh"
#include "tensor/tensor.h"
#include "histogram_kernel.cuh"

#define CUDA_CHECK(call) do {                                      \
  cudaError_t _e = (call);                                         \
  if (_e != cudaSuccess) {                                         \
    fprintf(stderr, "[CUDA] %s:%d: %s (%d)\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(_e), (int32_t)_e);  \
    std::exit(1);                                                  \
  }                                                                \
} while (0)

namespace kernel {
__global__ void histogram_kernel_v0(const int32_t* input,
                                    int32_t* output,
                                    int32_t low,
                                    int32_t high,
                                    int32_t ele_num) {
  int32_t thread_num = gridDim.x * blockDim.x;
  int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int32_t i = gid; i < ele_num; i += thread_num) {
    int32_t val = input[i];
    if (val >= low && val < high) {
      val -= low;
      atomicAdd(&output[val], 1);
    }
  }
}

// 每个block先在smem内维护一份局部histogram
// block内thread先更新s_histogram
// block再将结果合并到g_histogram
__global__ void histogram_kernel_v1(const int32_t* __restrict__ input,
                                    int32_t* __restrict__ output,
                                    int32_t low,
                                    int32_t high,
                                    int32_t ele_num) {
  int32_t tid = threadIdx.x;
  int32_t gid = blockIdx.x * blockDim.x + tid;
  int32_t thread_num = gridDim.x * blockDim.x;
  int32_t length = high - low;

  extern __shared__ int32_t smem_his[];
  for (int32_t i = tid; i < length; i += blockDim.x) {
    smem_his[i] = 0;
  }
  __syncthreads();

  for (int32_t idx = gid; idx < ele_num; idx += thread_num) {
    int32_t val = input[idx];
    if (val >= low && val < high) {
      val -= low;
      atomicAdd(&smem_his[val], 1);
    }
  }
  __syncthreads();

  for (int32_t i = tid; i < length; i += blockDim.x) {
    int32_t local_cnt = smem_his[i];
    if (local_cnt != 0) {
      atomicAdd(&output[i], local_cnt);
    }
  }
}

__global__ void histogram_kernel_v2(const int32_t* __restrict__ input,
                                    int32_t* __restrict__ output,
                                    int32_t low,
                                    int32_t high,
                                    int32_t ele_num,
                                    int32_t stride) {
  int32_t tid = threadIdx.x;
  int32_t gid = blockIdx.x * blockDim.x + tid;
  int32_t thread_num = gridDim.x * blockDim.x;
  int32_t warp_id = tid >> 5;
  int32_t warp_num = blockDim.x >> 5;
  int32_t lane_id = tid & 31;
  extern __shared__ int32_t smem_his[];

  int32_t length = high - low;
  int32_t total_bins = warp_num * stride;
  for (int i = tid; i < total_bins; i += blockDim.x) {
    smem_his[i] = 0;
  }
  __syncthreads();

  int32_t* warp_his = smem_his + warp_id * stride;
  for (int32_t idx = gid; idx < ele_num; idx += thread_num) {
    int32_t val = input[idx];
    if (val >= low && val < high) {
      val -= low;
      atomicAdd(&warp_his[val], 1);
    }
  }
  __syncthreads();

  for (int32_t i = lane_id; i < length; i += 32) {
    int32_t local_cnt = warp_his[i];
    if (local_cnt != 0) {
      atomicAdd(&output[i], local_cnt);
    }
  }
}

void histogram_kernel_cu(const tensor::Tensor &input,
                         tensor::Tensor &output,
                         para::histogram_para para,
                         void* stream){
  CHECK_EQ(input.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  int32_t block_num = 512;
  int32_t thread_num = 1024;
  int32_t warp_num = thread_num >> 5;
  dim3 grid(block_num);
  dim3 block(thread_num);

  int32_t ele_num = para.ele_num;
  int32_t low = para.low;
  int32_t high = para.high;
  int32_t length = high - low;
  int32_t stride = math_cu::AlignUp<int32_t>(length, 32);
  int32_t smem_size = warp_num * stride * sizeof(int32_t);

  int smem_size_max = 48 * 1024;
  if (smem_size > smem_size_max) {
    printf("smem_size: %d\n", smem_size);
    return;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    cudaEventRecord(start);
    // histogram_kernel_v1<<<grid, block, smem_size, stream_>>>(
    //   input.ptr<int32_t>(), output.ptr<int32_t>(), low, high, ele_num);
    histogram_kernel_v2<<<grid, block, smem_size, stream_>>>(
      input.ptr<int32_t>(), output.ptr<int32_t>(), low, high, ele_num, stride);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  } else {
    // histogram_kernel_v1<<<grid, block, smem_size>>>(
    //   input.ptr<int32_t>(), output.ptr<int32_t>(), low, high, ele_num);
    histogram_kernel_v2<<<grid, block, smem_size>>>(
      input.ptr<int32_t>(), output.ptr<int32_t>(), low, high, ele_num, stride);
  }

  // 立刻检查 launch 参数/配置错误（比如 invalid configuration）
  CUDA_CHECK(cudaPeekAtLastError());

  // 强制同步：如果 kernel 内部非法访存等运行时错误，这里会报出来
  CUDA_CHECK(cudaDeviceSynchronize());

  float gpu_time_ms = 0;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);
  std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
}
} // namespace kernel
