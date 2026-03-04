#include <random>
#include <vector>
#include <chrono>
#include <numeric>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                      \
  cudaError_t _e = (call);                                         \
  if (_e != cudaSuccess) {                                         \
    fprintf(stderr, "[CUDA] %s:%d: %s (%d)\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(_e), (int)_e);  \
    std::exit(1);                                                  \
  }                                                                \
} while (0)

const int BLOCK_SIZE = 1024;
const int N = 2048 * 7168;  // 512KB elements

float reduce_cpu(const std::vector<float> &data) {
  float sum = 0.0f;
  for (float val : data) {
    sum += val;
  }
  return sum;
}

// 1. 空闲thread较多, 随着stride变大, 工作thread越来越少
// 2. warp Divergence
// 3. bank conflict, 其实没有, 因为不在同一warp
//  eg:
//    stride = 1
//    sdata[0] += sdata[1];   tid0, tid32 不会冲突, 因为不在同一warp
__global__ void reduce_v0(float *g_idata, float *g_odata) {

  // blockDim固定为1024, 其余为gridDim
  // 第一轮reduce, 取得gridDim个中间结果 ->
  //  所以smem[blockDim], 求得当前block的归约结果
  // 第二轮reduce, 在gridDim个中间结果中, 得到最终结果
  //  此时分配1个block, gridDim个thread
  __shared__ float sdata[BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + tid;

  // 数据搬运&分块, 一维数组分成gridDim个block
  if (gid < N) {
    sdata[tid] = g_idata[gid];
  } else {
    sdata[tid] = 0.0f;
  }
  __syncthreads();

  // 每一轮, stride不断翻倍
  // [0, 1, 2, 3, 4, 5, 6, 7, 8]
  // i: 1   tid % 2 = 0 -> 0 + 1, 2 + 3, 4 + 5, 6 + 7
  // 0 + 1, 1, 2 + 3, 3, 4 + 5, 5, 6 + 7, 7, 8
  // i: 2   tid % 4 = 0 -> 0 + 2, 4 + 6
  // 0 + 1 + 2 + 3, 1, 2 + 3, 3, 4 + 5 + 6 + 7, 5, 6 + 7, 7, 8
  // i: 4   tid % 8 = 0 -> 0 + 4, 8
  // 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7
  // i: 8   tid % 16 = 0 -> 0 + 8
  // 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8
  for (unsigned int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[bid] = sdata[0];
  }
}

__global__ void reduce_v1(float* g_idata, float* g_odata, unsigned int n) {
  __shared__ float sdata[BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + tid;
  unsigned int thread_per_block = blockDim.x;

  // 传输数据的同时, 归约相邻原block的thread
  // eg: 512 * 1024
  //  block0: g[0~1023 + 0 * 1024] + g[0~1023 + 1 * 1024]
  //    b0 = b0 + b1   [0, 2047]
  //  block1: g[1024~2047 + 1 * 1024] + g[1024~2047 + 2 * 1024]
  //    b1 = b2 + b3   [2048, 4095]
  //    b2 = b4 + b5
  //          .
  //          .
  //          .
  //    b255 = b510 + b512
  //  所以实际分配的block, 是256

  // 第二次归约, 只分配1个block
  //  那么就归约相邻thread
  // eg: 1 * 512
  //  block0: g[0~255 + 0 * 256] + g[0~255 + 1 * 256]
  //    b00 = b00 + b01   [0, 511]
  // 所以第二次规约要分配256个thread
  unsigned int offset = gid + bid * thread_per_block;
  float a = offset < n ? g_idata[offset] : 0.0f;
  float b = offset + thread_per_block < n ?
                     g_idata[offset + thread_per_block] : 0.0f;
  sdata[tid] = a + b;
  __syncthreads();

  for (unsigned int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[bid] = sdata[0];
  }
}

__global__ void reduce_v2(float* g_idata, float* g_odata, unsigned int n) {
  __shared__ float sdata[BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + tid;
  unsigned int thread_per_block = blockDim.x;

  unsigned int offset = gid + bid * thread_per_block;
  float a = offset < n ? g_idata[offset] : 0.0f;
  float b = offset + thread_per_block < n ?
                     g_idata[offset + thread_per_block] : 0.0f;
  sdata[tid] = a + b;
  __syncthreads();

  // [0, 1, 2, 3, 4, 5, 6, 7]
  // tid0 += tid4
  // tid1 += tid5
  // tid2 += tid6
  // tid3 += tid7

  // tid0 += tid2
  // tid1 += tid3

  // tid0 += tid1

  // 保证了在循环前期, 整个warp会一起工作
  for (unsigned int stride = BLOCK_SIZE >> 1;
       stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[bid] = sdata[0];
  }
}

__device__ float warp_reduce(float val) {
  val += __shfl_down_sync(0xFFFFFFFF, val, 16);
  val += __shfl_down_sync(0xFFFFFFFF, val, 8);
  val += __shfl_down_sync(0xFFFFFFFF, val, 4);
  val += __shfl_down_sync(0xFFFFFFFF, val, 2);
  val += __shfl_down_sync(0xFFFFFFFF, val, 1);

  return val;
}

__global__ void reduce_v3(float* g_idata, float* g_odata, unsigned int n) {
  __shared__ float sdata[BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + tid;
  unsigned int thread_per_block = blockDim.x;

  unsigned int offset = gid + bid * thread_per_block;
  float a = offset < n ? g_idata[offset] : 0.0f;
  float b = offset + thread_per_block < n ?
                     g_idata[offset + thread_per_block] : 0.0f;
  sdata[tid] = a + b;
  __syncthreads();

  // 使用shfl进行最后一个warp的计算
  for (unsigned int stride = BLOCK_SIZE >> 1;
       stride > 16; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid < 32) {
    float val = warp_reduce(sdata[tid]);
    if (tid == 0) {
      g_odata[bid] = val;
    }
  }
}

__inline__ __device__ float block_reduce(float val) {
  const int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;

  // 每个warp内部进行归约
  val += __shfl_down_sync(0xFFFFFFFF, val, 16);
  val += __shfl_down_sync(0xFFFFFFFF, val, 8);
  val += __shfl_down_sync(0xFFFFFFFF, val, 4);
  val += __shfl_down_sync(0xFFFFFFFF, val, 2);
  val += __shfl_down_sync(0xFFFFFFFF, val, 1);

  // 同一个block的32个warp进行归约
  __shared__ float warpSums[32];
  if (lane == 0) {
    warpSums[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    val = (tid < blockDim.x / 32) ? warpSums[tid] : 0.0f;
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
  }

  return val;
}

// Q1: 不够鲁棒, 如果输入是127 * 1024, 第一轮归约可以正常进行,
//  第二轮需要分配127个thread, 无法正常计算
__global__ void reduce_v4(float* g_idata, float* g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + tid;
  unsigned int thread_per_block = blockDim.x;

  unsigned int offset = gid + bid * thread_per_block;
  float a = offset < n ? g_idata[offset] : 0.0f;
  float b = offset + thread_per_block < n ?
                     g_idata[offset + thread_per_block] : 0.0f;
  float sum = a + b;

  sum = block_reduce(sum);
  if (threadIdx.x == 0) {
    g_odata[bid] = sum;
  }
}

__device__ __forceinline__ float warp_reduce_v1(float val, unsigned mask) {
  val += __shfl_down_sync(mask, val, 16);
  val += __shfl_down_sync(mask, val, 8);
  val += __shfl_down_sync(mask, val, 4);
  val += __shfl_down_sync(mask, val, 2);
  val += __shfl_down_sync(mask, val, 1);
  return val;
}

__device__ __forceinline__ float block_reduce_v1(float val) {
  unsigned int tid = threadIdx.x;
  unsigned int lane_id = tid & 31;
  unsigned int warp_id = tid >> 5;
  unsigned int warp_num = (blockDim.x + 31) >> 5;

  __shared__ float warp_reduce[32];

  // 1. block内的所有warp归约
  unsigned int activate_lane = __activemask();
  val = warp_reduce_v1(val, activate_lane);
  if (lane_id == 0) {
    warp_reduce[warp_id] = val;
  }
  __syncthreads();

  // 2. 该block的第一个warp归约(第一轮后, 中间结果保存在这)
  if (warp_id == 0) {
    val = (lane_id < warp_num) ? warp_reduce[lane_id] : 0.0f;
    activate_lane = __activemask();
    val = warp_reduce_v1(val, activate_lane);
  }

  return val;
}

__global__ void reduce_v5(const float* __restrict__ g_idata,
                          float* __restrict__ g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  unsigned int gid = bid * blockDim.x + tid;
  unsigned int thread_per_block = blockDim.x;

  // 每个block处理两段连续区间
  unsigned int offset = gid + bid * thread_per_block;

  float a = offset < n ? g_idata[offset] : 0.0f;
  float b = offset + thread_per_block < n ?
                     g_idata[offset + thread_per_block] : 0.0f;
  float sum = a + b;

  // block内归约
  sum = block_reduce_v1(sum);

  if (tid == 0) {
    g_odata[bid] = sum;
  }
}

int main() {
  // int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // v0
  int num_blocks = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / 2; // v1

  std::vector<float> h_data(N);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(.0f, 1.f);
  for (int i = 0; i < N; ++i) {
    h_data[i] = dist_float(mt);
    // h_data[i] = 1;
  }

  // -------------------------------
  // CPU 计时开始
  auto cpu_start = std::chrono::high_resolution_clock::now();

  float cpu_result = reduce_cpu(h_data);

  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
  // CPU 计时结束
  // -------------------------------

  std::cout << "CPU result: " << cpu_result << std::endl;
  std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

  float gpu_result;
  float *d_final_result;
  float *d_data, *d_result;

  cudaMalloc(&d_data, N * sizeof(float));
  cudaMalloc(&d_result, num_blocks * sizeof(float));
  cudaMalloc(&d_final_result, 1 * sizeof(float));

  cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  // -------------------------------
  // GPU 计时开始 (CUDA Events)
  for (int i = 0; i < 10; ++i) {
    // reduce_v1<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    // reduce_v2<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    // reduce_v3<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    // reduce_v4<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    reduce_v5<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // reduce_v0<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result);
  // reduce_v1<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
  // reduce_v2<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
  // reduce_v3<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
  // reduce_v4<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);

  // 立刻检查 launch 参数/配置错误（比如 invalid configuration）
  CUDA_CHECK(cudaPeekAtLastError());

  // 强制同步：如果 kernel 内部非法访存等运行时错误，这里会报出来
  CUDA_CHECK(cudaDeviceSynchronize());
  // reduce_v0<<<1, BLOCK_SIZE>>>(d_result, d_final_result);
  // reduce_v1<<<1, BLOCK_SIZE>>>(d_result, d_final_result, N);
  // reduce_v2<<<1, BLOCK_SIZE>>>(d_result, d_final_result, N);
  // reduce_v3<<<1, BLOCK_SIZE>>>(d_result, d_final_result, N);
  // reduce_v4<<<1, BLOCK_SIZE>>>(d_result, d_final_result, N);

  // 立刻检查 launch 参数/配置错误（比如 invalid configuration）
  CUDA_CHECK(cudaPeekAtLastError());

  // 强制同步：如果 kernel 内部非法访存等运行时错误，这里会报出来
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned int res_N = 1;
  unsigned int cur_N = N;

  unsigned int max_ele_tmp = (N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
  float *d_tmp1, *d_tmp2;
  cudaMalloc(&d_tmp1, max_ele_tmp * sizeof(float));
  cudaMalloc(&d_tmp2, max_ele_tmp * sizeof(float));

  const float* in = d_data;
  float* out = d_tmp1;

  while (cur_N > res_N) {
    unsigned int block_num = (cur_N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
    reduce_v5<<<block_num, BLOCK_SIZE>>>(in, out, cur_N);

    cur_N = block_num;
    in = out;
    out = (out == d_tmp1) ? d_tmp2 : d_tmp1;
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // GPU 计时结束
  // -------------------------------

  std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;

  // cudaMemcpy(&gpu_result, d_final_result, sizeof(float),
  //            cudaMemcpyDeviceToHost);
  cudaMemcpy(&gpu_result, in, sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "GPU result: " << gpu_result << std::endl;

  if (abs(cpu_result - gpu_result) < 5e-1) {
    std::cout << "Result verified successfully!" << std::endl;
  } else {
    std::cout << "Result verification failed!" << std::endl;
  }

  // 清理资源
  cudaFree(d_data);
  cudaFree(d_result);
  cudaFree(d_final_result);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}