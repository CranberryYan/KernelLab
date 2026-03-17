#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// 以v3为base, 三个kernel进行融合

template <typename T>
void prefix_sum_CPU(T* in, T* out, int32_t len) {
  for (int i = 0; i < len; ++i) {
    out[i] = in[i];
  }
  for (int i = 1; i < len; ++i) {
    out[i] = out[i - 1] + out[i];
  }
}

template <typename T>
__global__ void scan_and_collect_sums(T* g_out, T* g_in, T* g_sums, int n) {
  cg::grid_group grid = cg::this_grid();
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int GRID_SIZE = gridDim.x;
  int BLOCK_SIZE = blockDim.x;
  int global_idx_base = bid * BLOCK_SIZE;

  extern __shared__ char smem[];
  T* vec = reinterpret_cast<T*>(smem);
  T* s_sums = reinterpret_cast<T*>(smem + BLOCK_SIZE * sizeof(T));

  if (global_idx_base + tid < n) {
    vec[tid] = g_in[global_idx_base + tid];
  } else {
    vec[tid] = 0;
  }
  __syncthreads();

  // stage1: block内prefix_sum_tmp
  // up-sweep
  int stride = 1;
  for (; stride < BLOCK_SIZE; stride <<= 1) {
    // tid + 1: 从1开始
    //  if tid: 从0开始, 0会与左边(-1)进行归约, -1 -> 未定义
    //  如果在下面添加限制, tid0固定空转, 浪费
    // stride: 半个区间
    // * 2: 整个区间
    // - 1: 将整个区间的末尾位置转换为0-based下标, index为改大区间的最右端点
    int index = (tid + 1) * stride * 2 - 1;
    if (index < BLOCK_SIZE) {
      vec[index] += vec[index - stride];
    }
    __syncthreads();
  }

  // down-sweep
  stride = BLOCK_SIZE >> 2;
  for (; stride > 0; stride >>= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index + stride < BLOCK_SIZE) {
      // 左边全部处理完了, 开始处理右边, 且左边(index)是规约好的
      vec[index + stride] += vec[index];
    }
    __syncthreads();
  }

  if (tid == BLOCK_SIZE - 1) {
    g_sums[bid] = vec[BLOCK_SIZE - 1];
  }
  grid.sync();

  // stage2: 求出g_sums的prefix_sum
  if (bid == 0) {
    if (tid < GRID_SIZE) {
      s_sums[tid] = g_sums[tid];
    } else {
      s_sums[tid] = static_cast<T>(0);
    }
    __syncthreads();

    // up-sweep
    int stride = 1;
    for (; stride < GRID_SIZE; stride <<= 1) {
      int index = (tid + 1) * stride * 2 - 1;
      if (index < GRID_SIZE) {
        s_sums[index] += s_sums[index - stride];
      }
      __syncthreads();
    }

    // dwon-sweep
    stride = GRID_SIZE >> 2;
    for (; stride > 0; stride >>= 1) {
      int index = (tid + 1) * stride * 2 - 1;
      if (index + stride < GRID_SIZE) {
        s_sums[index + stride] += s_sums[index];
      }
      __syncthreads();
    }

    if (tid < GRID_SIZE) {
      g_sums[tid] = s_sums[tid];
    }
  }
  grid.sync();

  // stage3: 将g_sums加到prefix_sum_tmp, 求出最终结果
  T add_val = (bid == 0) ? T(0) : g_sums[bid - 1];
  if (global_idx_base + tid < n) {
    g_out[global_idx_base + tid] = vec[tid] + add_val;
  }
}

int main() {
  // const int n = 1024 * 80;
  // tmp: 128
  // numBlocks: 128
  // cooperative launch too large: numBlocks=128, max=84

  const int n = 1024 * 64;
  const int thread_per_block = 1024;
  int numBlocks = (n + thread_per_block - 1) / thread_per_block;
  printf("numBlocks: %d\n", numBlocks);
  int tmp = 1;
  while (tmp < numBlocks) {
      tmp <<= 1;
  }
  numBlocks = tmp;
  printf("numBlocks: %d\n", numBlocks);

  const int32_t size = n * sizeof(int);
  const int32_t smem_size = thread_per_block * 2 * sizeof(int);

  int* h_in = reinterpret_cast<int*>(malloc(size));
  int* h_out = reinterpret_cast<int*>(malloc(size));
  int* h_res = reinterpret_cast<int*>(malloc(size));

  for (int i = 0; i < n; ++i) {
    h_in[i] = i % 10;
  }

  prefix_sum_CPU<int>(h_in, h_res, n);

  int* d_in, *d_out, *d_sums;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);
  cudaMalloc(&d_sums, numBlocks * sizeof(int));

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  // 检查设备是否支持cooperative launch
  int dev = 0;
  cudaSetDevice(dev);

  int supportsCoop = 0;
  cudaDeviceGetAttribute(&supportsCoop, cudaDevAttrCooperativeLaunch, dev);
  if (!supportsCoop) {
    printf("device does not support cooperative launch\n");
    return 1;
  }

  // 检查cooperative launch的block上限
  int maxBlocksPerSm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocksPerSm,
      scan_and_collect_sums<int>,
      thread_per_block,
      (int)smem_size);

  int smCount = 0;
  cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev);

  int maxCoopBlocks = maxBlocksPerSm * smCount;
  if (numBlocks > maxCoopBlocks) {
    printf("cooperative launch too large: numBlocks=%d, max=%d\n",
           numBlocks, maxCoopBlocks);
    return 1;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 预热
  for (int i = 0; i < 10; ++i) {
    void* args[] = {&d_out, &d_in, &d_sums, (void*)&n};
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)scan_and_collect_sums<int>,
        dim3(numBlocks), dim3(thread_per_block),
        args, smem_size, 0);
    if (err != cudaSuccess) {
      printf("launch failed: %s\n", cudaGetErrorString(err));
      return 0;
    }
  }
  cudaDeviceSynchronize();

  // 正式计时
  const int iters = 100000;

  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i) {
    void* args[] = {&d_out, &d_in, &d_sums, (void*)&n};
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)scan_and_collect_sums<int>,
        dim3(numBlocks), dim3(thread_per_block),
        args, smem_size, 0);
    if (err != cudaSuccess) {
      printf("launch failed: %s\n", cudaGetErrorString(err));
      return 0;
    }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  printf("total time = %.3f ms\n", ms);
  printf("avg kernel time = %.6f ms\n", ms / iters);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    if ((h_out[i] != h_res[i]) || (i < 3)) {
      printf("h_out[%d]: %d, h_res[%d]: %d\n", i, h_out[i], i, h_res[i]);
    }
  }

  free(h_in);
  free(h_out);
  free(h_res);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_sums);

  return 0;
}
