#include <cstdio>
#include <cuda_runtime.h>

// v4.3: Persistent CTAs + global task queue (single kernel).
// - A fixed number of CTAs persist and fetch tile_id via atomicAdd(queue).
// - Tile prefix dependency is resolved by DLB-like look-back state.
// - Epoch-tagged state avoids per-iteration memset over O(numTiles) arrays.

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
__global__ void prefix_sum_persistent_queue(
    T* g_out, const T* g_in, int n, int num_tiles,
    T* g_tile_agg, T* g_tile_prefix, int* g_state, int* g_next_tile, int epoch) {
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  extern __shared__ char smem[];
  T* vec = reinterpret_cast<T*>(smem);
  __shared__ int s_tile;
  __shared__ T s_excl;

  const int agg_ready = epoch * 2 + 1;
  const int prefix_ready = epoch * 2 + 2;

  while (true) {
    if (tid == 0) {
      s_tile = atomicAdd(g_next_tile, 1);
    }
    __syncthreads();

    int tile = s_tile;
    if (tile >= num_tiles) {
      break;
    }

    int base = tile * block_size;
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
      int rem = n - base;
      int last_valid = (rem < block_size) ? (rem - 1) : (block_size - 1);
      T tile_sum = static_cast<T>(0);
      if (last_valid >= 0) {
        tile_sum = vec[last_valid];
      }

      g_tile_agg[tile] = tile_sum;
      __threadfence();
      g_state[tile] = agg_ready;

      T excl = static_cast<T>(0);
      int look = tile - 1;
      volatile int* v_state = reinterpret_cast<volatile int*>(g_state);
      while (look >= 0) {
        int s = 0;
        do {
          s = v_state[look];
        } while (s < agg_ready);

        if (s >= prefix_ready) {
          excl += g_tile_prefix[look];
          break;
        } else {
          excl += g_tile_agg[look];
          --look;
        }
      }

      s_excl = excl;
      g_tile_prefix[tile] = excl + tile_sum;
      __threadfence();
      g_state[tile] = prefix_ready;
    }
    __syncthreads();

    T add = s_excl;
    if (base + tid < n) {
      g_out[base + tid] = vec[tid] + add;
    }
    __syncthreads();
  }
}

int main() {
  const int n = 1024 * 800;
  const int thread_per_block = 1024;
  const int numTiles = (n + thread_per_block - 1) / thread_per_block;
  printf("numTiles: %d\n", numTiles);

  int dev = 0;
  cudaSetDevice(dev);
  int smCount = 0;
  cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev);
  const int persistentBlocks = (numTiles < smCount * 2) ? numTiles : smCount * 2;
  printf("persistentBlocks: %d (smCount=%d)\n", persistentBlocks, smCount);

  const int32_t size = n * sizeof(int);
  const int32_t smem_size = thread_per_block * sizeof(int);

  int* h_in = reinterpret_cast<int*>(malloc(size));
  int* h_out = reinterpret_cast<int*>(malloc(size));
  int* h_res = reinterpret_cast<int*>(malloc(size));

  for (int i = 0; i < n; ++i) {
    h_in[i] = i % 10;
  }
  prefix_sum_CPU<int>(h_in, h_res, n);

  int *d_in, *d_out, *d_next_tile, *d_state;
  int *d_tile_agg, *d_tile_prefix;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);
  cudaMalloc(&d_tile_agg, numTiles * sizeof(int));
  cudaMalloc(&d_tile_prefix, numTiles * sizeof(int));
  cudaMalloc(&d_state, numTiles * sizeof(int));
  cudaMalloc(&d_next_tile, sizeof(int));

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  cudaMemset(d_state, 0, numTiles * sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int iters = 10000;
  int zero = 0;

  for (int w = 0; w < 10; ++w) {
    cudaMemcpy(d_next_tile, &zero, sizeof(int), cudaMemcpyHostToDevice);
    prefix_sum_persistent_queue<int><<<dim3(persistentBlocks), dim3(thread_per_block), smem_size>>>(
        d_out, d_in, n, numTiles, d_tile_agg, d_tile_prefix, d_state, d_next_tile, w + 1);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i) {
    cudaMemcpy(d_next_tile, &zero, sizeof(int), cudaMemcpyHostToDevice);
    prefix_sum_persistent_queue<int><<<dim3(persistentBlocks), dim3(thread_per_block), smem_size>>>(
        d_out, d_in, n, numTiles, d_tile_agg, d_tile_prefix, d_state, d_next_tile, i + 100);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  printf("total time (v4.3 persistent+queue single-kernel) = %.3f ms\n", ms);
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
  cudaFree(d_tile_agg);
  cudaFree(d_tile_prefix);
  cudaFree(d_state);
  cudaFree(d_next_tile);

  return 0;
}
