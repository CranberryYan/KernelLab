#include <cstdio>
#include <cuda_runtime.h>

// v5: 三 kernel 拆分，去掉 cooperative launch 与 grid.sync()
// K1 块内 inclusive scan -> g_partial + g_sums[bid]
// K2 单 block 对 g_sums 做 inclusive scan（与 v4 stage2 同构）
// K3 加块间前缀到 g_partial -> g_out

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
__global__ void scan_blocks(T* g_partial, T* g_sums, const T* g_in, int n) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int BLOCK_SIZE = blockDim.x;
  int global_idx_base = bid * BLOCK_SIZE;

  extern __shared__ char smem[];
  T* vec = reinterpret_cast<T*>(smem);

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

  if (tid == BLOCK_SIZE - 1) {
    g_sums[bid] = vec[BLOCK_SIZE - 1];
  }
  if (global_idx_base + tid < n) {
    g_partial[global_idx_base + tid] = vec[tid];
  }
}

template <typename T>
__global__ void scan_block_sums(T* g_sums, int GRID_SIZE) {
  int tid = threadIdx.x;

  extern __shared__ char smem[];
  T* s_sums = reinterpret_cast<T*>(smem);

  if (tid < GRID_SIZE) {
    s_sums[tid] = g_sums[tid];
  } else {
    s_sums[tid] = static_cast<T>(0);
  }
  __syncthreads();

  int stride = 1;
  for (; stride < GRID_SIZE; stride <<= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < GRID_SIZE) {
      s_sums[index] += s_sums[index - stride];
    }
    __syncthreads();
  }

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

template <typename T>
__global__ void add_block_prefix(T* g_out, const T* g_partial, const T* g_sums,
                                int n) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int BLOCK_SIZE = blockDim.x;
  int global_idx_base = bid * BLOCK_SIZE;

  T add_val = (bid == 0) ? T(0) : g_sums[bid - 1];
  if (global_idx_base + tid < n) {
    g_out[global_idx_base + tid] = g_partial[global_idx_base + tid] + add_val;
  }
}

int main() {
  const int n = 1023 * 799;
  const int thread_per_block = 1024;
  int numBlocks = (n + thread_per_block - 1) / thread_per_block;
  printf("numBlocks: %d\n", numBlocks);
  int tmp = 1;
  while (tmp < numBlocks) {
      tmp <<= 1;
  }
  numBlocks = tmp;
  printf("numBlocks: %d\n", numBlocks);

  if (numBlocks > thread_per_block) {
    printf("v5: numBlocks (%d) > thread_per_block (%d), K2 needs extension\n",
           numBlocks, thread_per_block);
    return 1;
  }

  const int32_t size = n * sizeof(int);
  const int32_t smem_blocks = thread_per_block * sizeof(int);
  const int32_t smem_sums = thread_per_block * sizeof(int);

  int* h_in = reinterpret_cast<int*>(malloc(size));
  int* h_out = reinterpret_cast<int*>(malloc(size));
  int* h_res = reinterpret_cast<int*>(malloc(size));

  for (int i = 0; i < n; ++i) {
    h_in[i] = i % 10;
  }

  prefix_sum_CPU<int>(h_in, h_res, n);

  int *d_in, *d_out, *d_sums, *d_partial;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);
  cudaMalloc(&d_sums, numBlocks * sizeof(int));
  cudaMalloc(&d_partial, size);

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int iters = 10000;

  for (int w = 0; w < 10; ++w) {
    scan_blocks<int><<<dim3(numBlocks), dim3(thread_per_block), smem_blocks>>>(
        d_partial, d_sums, d_in, n);
    scan_block_sums<int><<<dim3(1), dim3(thread_per_block), smem_sums>>>(
        d_sums, numBlocks);
    add_block_prefix<int><<<dim3(numBlocks), dim3(thread_per_block)>>>(
        d_out, d_partial, d_sums, n);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i) {
    scan_blocks<int><<<dim3(numBlocks), dim3(thread_per_block), smem_blocks>>>(
        d_partial, d_sums, d_in, n);
    scan_block_sums<int><<<dim3(1), dim3(thread_per_block), smem_sums>>>(
        d_sums, numBlocks);
    add_block_prefix<int><<<dim3(numBlocks), dim3(thread_per_block)>>>(
        d_out, d_partial, d_sums, n);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  printf("total time (3 kernels) = %.3f ms\n", ms);
  printf("avg per iteration = %.6f ms\n", ms / iters);

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
  cudaFree(d_partial);

  return 0;
}
