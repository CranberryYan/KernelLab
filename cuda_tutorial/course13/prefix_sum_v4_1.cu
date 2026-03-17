#include <cstdio>
#include <cuda_runtime.h>

// v4.1：单 kernel inclusive prefix sum（链式 scan / decoupled 依赖）
// - 无 cooperative launch、无 grid.sync()
// - 块 b 等待块 b-1 写完「块 b 起始处的 exclusive 前缀」后再写本块输出
//
// 测时建议（main 中 [A]/[B]）：
// - [A] 每轮 cudaMemset(d_ready)+kernel：与「每次独立调用」一致，含真实链式握手与 memset 成本，用于和 v4/v5 公平对比。
// - [B] 仅首次 memset 后连跑 kernel：同输入重复时 ready 恒为 1，自旋被旁路，仅反映「热循环」下界，不能代表通用单次调用。

template <typename T>
void prefix_sum_CPU(T* in, T* out, int32_t len) {
  for (int i = 0; i < len; ++i) {
    out[i] = in[i];
  }
  for (int i = 1; i < len; ++i) {
    out[i] = out[i - 1] + out[i];
  }
}

// g_prefix[b] = 全局 exclusive 前缀：下标 < block b 起始位置 的元素之和（块 b 从 g_prefix[b] 起加）
// g_ready[b] = 1 表示块 b 已写完 g_prefix[b+1] 且可被后续块使用
template <typename T>
__global__ void prefix_sum_inclusive_chained(T* g_out, const T* g_in, int n,
                                           T* g_prefix, int* g_ready) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_blocks = gridDim.x;
  int BLOCK_SIZE = blockDim.x;
  int global_idx_base = bid * BLOCK_SIZE;

  extern __shared__ char smem[];
  T* vec = reinterpret_cast<T*>(smem);
  __shared__ T s_excl;

  if (global_idx_base + tid < n) {
    vec[tid] = g_in[global_idx_base + tid];
  } else {
    vec[tid] = static_cast<T>(0);
  }
  __syncthreads();

  int stride = 1;
  for (; stride < BLOCK_SIZE; stride <<= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < BLOCK_SIZE) {
      vec[index] += vec[index - stride];
    }
    __syncthreads();
  }

  stride = BLOCK_SIZE >> 2;
  for (; stride > 0; stride >>= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index + stride < BLOCK_SIZE) {
      vec[index + stride] += vec[index];
    }
    __syncthreads();
  }

  if (tid == 0) {
    T excl;
    if (bid == 0) {
      excl = static_cast<T>(0);
    } else {
      volatile int* ready = reinterpret_cast<volatile int*>(g_ready);
      while (ready[bid - 1] == 0) {
      }
      excl = g_prefix[bid];
    }
    s_excl = excl;
  }
  __syncthreads();

  T add = s_excl;
  if (global_idx_base + tid < n) {
    g_out[global_idx_base + tid] = vec[tid] + add;
  }
  __syncthreads();

  if (tid == 0) {
    int last_valid = -1;
    if (global_idx_base < n) {
      int rem = n - global_idx_base;
      last_valid = (rem < BLOCK_SIZE) ? (rem - 1) : (BLOCK_SIZE - 1);
    }
    T tile_sum = static_cast<T>(0);
    if (last_valid >= 0) {
      tile_sum = vec[last_valid];
    }
    T excl = s_excl;
    T next_prefix = excl + tile_sum;
    if (bid + 1 < num_blocks) {
      g_prefix[bid + 1] = next_prefix;
    }
    __threadfence();
    g_ready[bid] = 1;
  }
}

int main() {
  const int n = 1024 * 800;
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
  const int32_t smem_chained = thread_per_block * sizeof(int);

  int* h_in = reinterpret_cast<int*>(malloc(size));
  int* h_out = reinterpret_cast<int*>(malloc(size));
  int* h_res = reinterpret_cast<int*>(malloc(size));

  for (int i = 0; i < n; ++i) {
    h_in[i] = i % 10;
  }

  prefix_sum_CPU<int>(h_in, h_res, n);

  int *d_in, *d_out, *d_prefix, *d_ready;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);
  cudaMalloc(&d_prefix, numBlocks * sizeof(int));
  cudaMalloc(&d_ready, numBlocks * sizeof(int));

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int iters = 1000;

  // 预热与 [A] 一致：每轮清零 d_ready，避免与计时路径不一致
  for (int w = 0; w < 10; ++w) {
    cudaMemset(d_ready, 0, numBlocks * sizeof(int));
    prefix_sum_inclusive_chained<int><<<dim3(numBlocks), dim3(thread_per_block),
                                      smem_chained>>>(d_out, d_in, n, d_prefix,
                                                      d_ready);
  }
  cudaDeviceSynchronize();

  // [A] 主指标：每轮 memset + kernel（真实链式依赖 + API 成本）
  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i) {
    cudaMemset(d_ready, 0, numBlocks * sizeof(int));
    prefix_sum_inclusive_chained<int><<<dim3(numBlocks), dim3(thread_per_block),
                                      smem_chained>>>(d_out, d_in, n, d_prefix,
                                                      d_ready);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float ms_a = 0.0f;
  cudaEventElapsedTime(&ms_a, start, stop);
  printf("[A] per-invocation reset: total %.3f ms, avg %.6f ms/iter "
         "(cudaMemset + kernel; use for vs v4/v5)\n",
         ms_a, ms_a / iters);

  // [B] 参考：仅首次清零后连跑 — 同输入下 g_ready 保持为 1，自旋几乎不触发
  cudaMemset(d_ready, 0, numBlocks * sizeof(int));
  prefix_sum_inclusive_chained<int><<<dim3(numBlocks), dim3(thread_per_block),
                                    smem_chained>>>(d_out, d_in, n, d_prefix,
                                                    d_ready);
  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i) {
    prefix_sum_inclusive_chained<int><<<dim3(numBlocks), dim3(thread_per_block),
                                      smem_chained>>>(d_out, d_in, n, d_prefix,
                                                      d_ready);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float ms_b = 0.0f;
  cudaEventElapsedTime(&ms_b, start, stop);
  printf("[B] hot repeat (no per-iter memset): total %.3f ms, avg %.6f ms/iter "
         "(NOT general single-call cost)\n",
         ms_b, ms_b / iters);

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
  cudaFree(d_prefix);
  cudaFree(d_ready);

  return 0;
}
