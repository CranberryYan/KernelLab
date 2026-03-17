// v0带宽利用率: 62.67%
// v0内存吞吐量: 144.33GB/s
// v1: 解决 warp divergent
// v1带宽利用率: 86.46%
// v1内存吞吐量: 199.35GB/s
// v2: 解决 bank conflict
// v2带宽利用率: 89.95%
// v2内存吞吐量: 207.42GB/s
// v3: 展开最后一个warp的循环
// v3带宽利用率: 92.26%
// v4内存吞吐量: 674.85GB/s(比较接近760GB/s的理论值)


#include <cub/cub.cuh>
#include "math_utils.cuh"
#include "tensor/tensor.h"
#include "reduce_kernel.cuh"

namespace kernel {
__global__ void reduce_kernel_v0(const float* input, float* output) {
  // 256个thread
  // 256 * 32/8 = 1024Bytes -> 1kb
  // 3080: 单个SM的L1 cache 128kb
  extern __shared__ float smem[];
  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // gmem -> smem
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

  // gmem -> smem
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

  // gmem -> smem
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

  // [1, 2, 3, 4, 5, 6, 7, 8]
  // [1 + 5, 2 + 6, 3 + 7, 4 + 8, 5, 6, 7 ,8]
  // [1 + 5 + 2 + 6, 2 + 6, 3 + 7 + 4 + 8, 4 + 8, 5, 6, 7, 8]
  // [1 + 5 + 2 + 6 + 3 + 7 + 4 + 8, ...]
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

// volatile: 告诉编译器 “不要优化这段内存访问”，每次都要真的访问内存
//  为了确保: 每次读取 smem[tid + x] 都是真的从共享内存中读取，而不是从寄存器中读取可能旧的值
__device__ void unroll_last_warp(volatile float *smem, uint32_t tid) {
  smem[tid] += smem[tid + 32];
  smem[tid] += smem[tid + 16];
  smem[tid] += smem[tid + 8];
  smem[tid] += smem[tid + 4];
  smem[tid] += smem[tid + 2];
  smem[tid] += smem[tid + 1];
}

__global__ void reduce_kernel_v3(const float* input, float* output) {
  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // gmem -> smem
  extern __shared__ float smem[]; // 外部已经定义smem的大小(注意要使用extern关键字)
  smem[tid] = input[gid];
  __syncthreads();

  // [1, 2, 3, 4, 5, 6, 7, 8]
  // [1 + 5, 2 + 6, 3 + 7, 4 + 6, 5, 6, 7, 8]
  for (int i = blockDim.x / 2; i > 32; i >>= 1) {
    if (tid < i) {
      smem[tid] += smem[tid + i];
    }
    __syncthreads();
  }

  // unroll last warp
  if (tid < 32) {
    unroll_last_warp(smem, tid);
  }

  // 每个block的第0个thread, 将结果smem -> gmem
  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
  __syncthreads();
}

template <typename T, uint32_t mode>
struct ReduceOp;

// mode=0: sum
template <typename T>
struct ReduceOp<T, 0> {
  __device__ __forceinline__ static T identity() { return T(0); }
  __device__ __forceinline__ static T apply(T a, T b) { return a + b; }
};

// mode=1: max
template <>
struct ReduceOp<float, 1> {
  __device__ __forceinline__ static float identity() { return -INFINITY; }
  __device__ __forceinline__ static float apply(float a, float b) { return fmaxf(a, b); }
};
template <>
struct ReduceOp<int32_t, 1> {
  __device__ __forceinline__ static int32_t identity() { return INT32_MIN; }
  __device__ __forceinline__ static int32_t apply(int32_t a, int32_t b) { return (a > b) ? a : b; }
};
template <>
struct ReduceOp<uint32_t, 1> {
  __device__ __forceinline__ static uint32_t identity() { return 0u; }
  __device__ __forceinline__ static uint32_t apply(uint32_t a, uint32_t b) { return (a > b) ? a : b; }
};

// mode=2: min
template <>
struct ReduceOp<float, 2> {
  __device__ __forceinline__ static float identity() { return INFINITY; }
  __device__ __forceinline__ static float apply(float a, float b) { return fminf(a, b); }
};
template <>
struct ReduceOp<int32_t, 2> {
  __device__ __forceinline__ static int32_t identity() { return INT32_MAX; }
  __device__ __forceinline__ static int32_t apply(int32_t a, int32_t b) { return (a < b) ? a : b; }
};
template <>
struct ReduceOp<uint32_t, 2> {
  __device__ __forceinline__ static uint32_t identity() { return 0xFFFFFFFFu; }
  __device__ __forceinline__ static uint32_t apply(uint32_t a, uint32_t b) { return (a < b) ? a : b; }
};

template<typename T, uint32_t reduce_mode>
__device__ __forceinline__ T warp_reduce_v1(T val, unsigned mask) {
  #pragma unroll
  for (int stride = 16; stride >= 1; stride >>= 1) {
    T other = __shfl_down_sync(mask, val, stride);
    val = ReduceOp<T, reduce_mode>::apply(val, other);
  }
  return val;
}

template<typename T, uint32_t reduce_mode>
__device__ __forceinline__ T block_reduce_v1(T val) {
  uint32_t tid = threadIdx.x;
  uint32_t lane = tid & 31;
  uint32_t warp_id = tid >> 5;
  uint32_t warp_num = (blockDim.x + 31) >> 5;

  __shared__ T warp_sums[32];

  uint32_t m = __activemask();
  val = warp_reduce_v1<T, reduce_mode>(val, m);

  if (lane == 0) warp_sums[warp_id] = val;
  __syncthreads();

  if (warp_id == 0) {
    val = (lane < warp_num) ?
          warp_sums[lane] :
          ReduceOp<T, reduce_mode>::identity();
    unsigned m0 = __activemask();
    val = warp_reduce_v1<T, reduce_mode>(val, m0);
  }
  return val;
}

template<typename T, uint32_t reduce_mode>
__global__ void reduce_kernel_v4(const T* __restrict__ input,
                                 T* __restrict__ output,
                                 unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  unsigned int gid = bid * blockDim.x + tid;
  unsigned int thread_per_block = blockDim.x;

  // 每个block处理两段连续区间
  unsigned int offset = gid + bid * thread_per_block;

  // 0: sum   1: max   2: min
  T a = (offset < n)?
        input[offset]:
        ReduceOp<T, reduce_mode>::identity();
  T b = (offset + thread_per_block < n) ?
        input[offset + thread_per_block] :
        ReduceOp<T, reduce_mode>::identity();
  T val = ReduceOp<T, reduce_mode>::apply(a, b);

  // block内归约
  val = block_reduce_v1<T, reduce_mode>(val);

  if (tid == 0) {
    output[bid] = val;
  }
}

#if 1
template <typename T>
static inline void launch_reduce(cudaStream_t s, dim3 grid, dim3 block,
                                 const T* in, T* out, unsigned int n,
                                 uint32_t reduce_mode) {
  switch (reduce_mode) {
    case 0: reduce_kernel_v4<T, 0><<<grid, block, 0, s>>>(in, out, n); break;
    case 1: reduce_kernel_v4<T, 1><<<grid, block, 0, s>>>(in, out, n); break;
    case 2: reduce_kernel_v4<T, 2><<<grid, block, 0, s>>>(in, out, n); break;
    default: CHECK(false) << "Unsupported reduce_mode: " << reduce_mode;
  }
}

template <typename T>
void reduce_kernel_cu_typed(const tensor::Tensor& input,
                            tensor::Tensor& output,
                            para::reduce_para para,
                            void* stream) {
  CHECK_EQ(input.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  CHECK_EQ(static_cast<int>(output.data_type()),
           static_cast<int>(input.data_type()));

  unsigned int res_N = para.after_reduce_num;
  unsigned int cur_N = para.ele_num;

  uint32_t thread_num;
  if (cur_N <= 1024 * 256) thread_num = 256;
  else if (cur_N <= 2048 * 512) thread_num = 512;
  else thread_num = 1024;

  CHECK_EQ(thread_num % 32, 0);
  CHECK_LE(thread_num, 1024);

  unsigned int max_ele_tmp = math_cu::CeilDiv<uint32_t>(cur_N, 2 * thread_num);

  T* d_tmp1 = nullptr;
  T* d_tmp2 = nullptr;
  cudaMalloc(&d_tmp1, max_ele_tmp * sizeof(T));
  cudaMalloc(&d_tmp2, max_ele_tmp * sizeof(T));

  const T* in = input.ptr<T>();
  T* out = d_tmp1;

  cudaStream_t stream_ = stream ? reinterpret_cast<cudaStream_t>(stream) : 0;

  while (cur_N > res_N) {
    uint32_t block_num = math_cu::CeilDiv<uint32_t>(cur_N, 2 * thread_num);
    dim3 grid(block_num);
    dim3 block(thread_num);

    launch_reduce<T>(stream_, grid, block, in, out, cur_N, para.reduce_mode);

    cur_N = block_num;
    in = out;
    out = (out == d_tmp1) ? d_tmp2 : d_tmp1;
  }

  cudaMemcpyAsync(output.ptr<T>(), in, res_N * sizeof(T),
                  cudaMemcpyDeviceToDevice, stream_);

  cudaFree(d_tmp1);
  cudaFree(d_tmp2);
}

void reduce_kernel_cu(const tensor::Tensor& input,
                      tensor::Tensor& output,
                      para::reduce_para para,
                      void* stream) {
  switch (input.data_type()) {
    case base::DataType::kDataTypeFp32:
      reduce_kernel_cu_typed<float>(input, output, para, stream);
      break;
    case base::DataType::kDataTypeInt32:
      reduce_kernel_cu_typed<int32_t>(input, output, para, stream);
      break;
    case base::DataType::kDataTypeUInt32:
      reduce_kernel_cu_typed<uint32_t>(input, output, para, stream);
      break;
    default:
      CHECK(false) << "Unsupported data type: " << int(input.data_type());
  }
}

#if 0
void reduce_kernel_cu(const tensor::Tensor &input,
                      tensor::Tensor &output,
                      para::reduce_para para,
                      void* stream) {
  CHECK_EQ(input.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  unsigned int res_N = para.after_reduce_num;
  unsigned int cur_N = para.ele_num;

  uint32_t thread_num = 256;
  uint32_t block_num = math_cu::CeilDiv<uint32_t>(cur_N, 2*thread_num);
  if (cur_N <= 256 * 1024) thread_num = 256;
  else if (cur_N <= 1024 * 1024) thread_num = 512;
  else thread_num = 1024;
  block_num = math_cu::CeilDiv<uint32_t>(cur_N, 2*thread_num);
  printf("block_num: %d, thread_num: %d\n", block_num, thread_num);

  unsigned int max_ele_tmp = math_cu::CeilDiv<uint32_t>(cur_N, 2*thread_num);
  printf("res_N: %d, cur_N: %d\n", res_N, cur_N);

  if (input.data_type() == base::DataType::kDataTypeFp32) {
    T = DataTypeToCppT<base::DataType::kDataTypeFp32>;
  } else if (input.data_type() == base::DataType::kDataTypeInt32) {
    T = DataTypeToCppT<base::DataType::kDataTypeInt32>;
  } else if (input.data_type() == base::DataType::kDataTypeUInt32) {
    T = DataTypeToCppT<base::DataType::kDataTypeUInt32>;
  }

  T* d_tmp1 = nullptr;
  T* d_tmp2 = nullptr;
  cudaMalloc(&d_tmp1, max_ele_tmp * sizeof(T));
  cudaMalloc(&d_tmp2, max_ele_tmp * sizeof(T));

  const T* in = input.ptr<T>();
  T* out = d_tmp1;

  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    while (cur_N > res_N) {
      block_num = math_cu::CeilDiv<uint32_t>(cur_N, 2*thread_num);
      printf("while block_num: %d, thread_num: %d\n", block_num, thread_num);

      CHECK_EQ(thread_num % 32, 0);
      CHECK_LE(thread_num, 1024);

      dim3 grid(block_num);
      dim3 block(thread_num);
      if (para.reduce_mode == 0) {
        reduce_kernel_v4<T, 0><<<grid, block, 0, stream_>>>(
          in, out, cur_N);
      } else if (para.reduce_mode == 1) {
        reduce_kernel_v4<T, 1><<<grid, block, 0, stream_>>>(
          in, out, cur_N);
      } else if (para.reduce_mode == 2) {
        reduce_kernel_v4<T, 2><<<grid, block, 0, stream_>>>(
          in, out, cur_N);
      }

      cur_N = block_num;
      in = out;
      out = (out == d_tmp1) ? d_tmp2 : d_tmp1;
    }
    // 最终结果在 in[0..res_N-1]，拷回 output tensor
    cudaMemcpyAsync(output.ptr<T>(), in, res_N * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream_);
  } else {
    while (cur_N > res_N) {
      block_num = math_cu::CeilDiv<uint32_t>(cur_N, 2*thread_num);
      printf("while block_num: %d, thread_num: %d\n", block_num, thread_num);

      CHECK_EQ(thread_num % 32, 0);
      CHECK_LE(thread_num, 1024);

      dim3 grid(block_num);
      dim3 block(thread_num);
      if (para.reduce_mode == 0) {
        reduce_kernel_v4<T, 0><<<grid, block>>>(in, out, cur_N);
      } else if (para.reduce_mode == 1) {
        reduce_kernel_v4<T, 1><<<grid, block>>>(in, out, cur_N);
      } else if (para.reduce_mode == 2) {
        reduce_kernel_v4<T, 2><<<grid, block>>>(in, out, cur_N);
      }

      cur_N = block_num;
      in = out;
      out = (out == d_tmp1) ? d_tmp2 : d_tmp1;
    }
    // 最终结果在 in[0..res_N-1]，拷回 output tensor
    cudaMemcpyAsync(output.ptr<T>(), in, res_N * sizeof(T),
                    cudaMemcpyDeviceToDevice);
  }

  cudaFree(d_tmp1);
  cudaFree(d_tmp2);
}
#endif
#endif
} // namespace kernel
