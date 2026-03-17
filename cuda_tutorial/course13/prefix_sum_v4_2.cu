#include <cstdio>
#include <cuda_runtime.h>

// v4.2: single-kernel inclusive scan with DLB-like look-back.
// - No cooperative launch, no grid.sync.
// - Each block publishes tile aggregate first, then resolves its carry via look-back.
// - Epoch-tagged state avoids per-iteration cudaMemset.

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
__global__ void prefix_sum_inclusive_dlb(
    T* g_out, const T* g_in, int n,
    T* g_block_agg, T* g_block_prefix, int* g_state, int epoch) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int block_size = blockDim.x;
  int base = bid * block_size;

  extern __shared__ char smem[];
  T* vec = reinterpret_cast<T*>(smem);
  __shared__ T s_excl;

  if (base + tid < n) {
    vec[tid] = g_in[base + tid];
  } else {
    vec[tid] = static_cast<T>(0);
  }
  __syncthreads();

  int stride = 1;
  for (; stride < block_size; stride <<= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < block_size) {
      vec[index] += vec[index - stride];
    }
    __syncthreads();
  }

  stride = block_size >> 2;
  for (; stride > 0; stride >>= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index + stride < block_size) {
      vec[index + stride] += vec[index];
    }
    __syncthreads();
  }

  if (tid == 0) {
    int last_valid = -1;
    if (base < n) {
      int rem = n - base;
      last_valid = (rem < block_size) ? (rem - 1) : (block_size - 1);
    }
    T tile_sum = static_cast<T>(0);
    if (last_valid >= 0) {
      tile_sum = vec[last_valid];
    }

    const int agg_ready = epoch * 2 + 1;
    const int prefix_ready = epoch * 2 + 2;

    g_block_agg[bid] = tile_sum;
    __threadfence();
    g_state[bid] = agg_ready;

    T excl = static_cast<T>(0);
    int look = bid - 1;
    volatile int* v_state = reinterpret_cast<volatile int*>(g_state);
    while (look >= 0) {
      int s = 0;
      do {
        s = v_state[look];
      } while (s < agg_ready);

      if (s >= prefix_ready) {
        excl += g_block_prefix[look];
        break;
      } else {
        excl += g_block_agg[look];
        --look;
      }
    }
    s_excl = excl;

    g_block_prefix[bid] = excl + tile_sum;
    __threadfence();
    g_state[bid] = prefix_ready;
  }
  __syncthreads();

  T add = s_excl;
  if (base + tid < n) {
    g_out[base + tid] = vec[tid] + add;
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
  const int32_t smem_size = thread_per_block * sizeof(int);

  int* h_in = reinterpret_cast<int*>(malloc(size));
  int* h_out = reinterpret_cast<int*>(malloc(size));
  int* h_res = reinterpret_cast<int*>(malloc(size));

  for (int i = 0; i < n; ++i) {
    h_in[i] = i % 10;
  }
  prefix_sum_CPU<int>(h_in, h_res, n);

  int *d_in, *d_out, *d_state;
  int *d_block_agg, *d_block_prefix;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);
  cudaMalloc(&d_block_agg, numBlocks * sizeof(int));
  cudaMalloc(&d_block_prefix, numBlocks * sizeof(int));
  cudaMalloc(&d_state, numBlocks * sizeof(int));

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  cudaMemset(d_state, 0, numBlocks * sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int iters = 1000;

  for (int w = 0; w < 10; ++w) {
    prefix_sum_inclusive_dlb<int><<<dim3(numBlocks), dim3(thread_per_block), smem_size>>>(
        d_out, d_in, n, d_block_agg, d_block_prefix, d_state, w + 1);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i) {
    prefix_sum_inclusive_dlb<int><<<dim3(numBlocks), dim3(thread_per_block), smem_size>>>(
        d_out, d_in, n, d_block_agg, d_block_prefix, d_state, i + 100);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  printf("total time (v4.2 single-kernel DLB-like) = %.3f ms\n", ms);
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
  cudaFree(d_block_agg);
  cudaFree(d_block_prefix);
  cudaFree(d_state);

  return 0;
}
