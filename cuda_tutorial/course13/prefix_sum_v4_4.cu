#include <cstdio>
#include <cuda_runtime.h>

template <typename T>
void prefix_sum_CPU(T* in, T* out, int32_t len) {
  for (int i = 0; i < len; ++i) {
    out[i] = in[i];
  }
  for (int i = 1; i < len; ++i) {
    out[i] = out[i - 1] + out[i];
  }
}

// 单kernel + DLB(Decoupled Look-Back)的inclusive scan
//  
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void prefix_sum_inclusive_dlb_optimized(
    T* g_out, const T* g_in, int n,
    T* g_tile_agg, T* g_tile_prefix, int* g_state, int epoch) {
  static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be warp-aligned");
  constexpr int WARP_SIZE = 32;
  constexpr int WARP_COUNT = BLOCK_THREADS / WARP_SIZE;
  constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

  const int tid = threadIdx.x;
  const int lane = tid & (WARP_SIZE - 1);
  const int warp = tid / WARP_SIZE;
  const int tile = blockIdx.x;
  const int tile_base = tile * TILE_ITEMS;

  __shared__ T warp_inclusive[WARP_COUNT];
  __shared__ T s_tile_sum;
  __shared__ T s_tile_exclusive;

  T vals[ITEMS_PER_THREAD];
  T local_prefix[ITEMS_PER_THREAD];

  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int idx = tile_base + tid * ITEMS_PER_THREAD + i;
    vals[i] = (idx < n) ? g_in[idx] : T(0);
  }

  T thread_sum = T(0);
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    thread_sum += vals[i];
    local_prefix[i] = thread_sum;
  }

  T warp_scan = thread_sum;
  #pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    T nbor = __shfl_up_sync(0xffffffff, warp_scan, offset);
    if (lane >= offset) {
      warp_scan += nbor;
    }
  }

  if (lane == WARP_SIZE - 1) {
    warp_inclusive[warp] = warp_scan;
  }
  __syncthreads();

  if (warp == 0) {
    T x = (lane < WARP_COUNT) ? warp_inclusive[lane] : T(0);
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
      T nbor = __shfl_up_sync(0xffffffff, x, offset);
      if (lane >= offset) {
        x += nbor;
      }
    }
    if (lane < WARP_COUNT) {
      warp_inclusive[lane] = x;
    }
  }
  __syncthreads();

  T warp_exclusive = (warp == 0) ? T(0) : warp_inclusive[warp - 1];
  T thread_exclusive = warp_exclusive + (warp_scan - thread_sum);

  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    local_prefix[i] += thread_exclusive;
  }

  if (tid == BLOCK_THREADS - 1) {
    s_tile_sum = local_prefix[ITEMS_PER_THREAD - 1];
  }
  __syncthreads();

  if (tid == 0) {
    const int agg_ready = epoch * 2 + 1;
    const int pref_ready = epoch * 2 + 2;

    g_tile_agg[tile] = s_tile_sum;
    __threadfence();
    g_state[tile] = agg_ready;

    T carry = T(0);
    int look = tile - 1;
    volatile int* v_state = reinterpret_cast<volatile int*>(g_state);
    while (look >= 0) {
      int s = 0;
      do {
        s = v_state[look];
      } while (s < agg_ready);

      if (s >= pref_ready) {
        carry += g_tile_prefix[look];
        break;
      } else {
        carry += g_tile_agg[look];
        --look;
      }
    }

    s_tile_exclusive = carry;
    g_tile_prefix[tile] = carry + s_tile_sum;
    __threadfence();
    g_state[tile] = pref_ready;
  }
  __syncthreads();

  T add = s_tile_exclusive;
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int idx = tile_base + tid * ITEMS_PER_THREAD + i;
    if (idx < n) {
      g_out[idx] = local_prefix[i] + add;
    }
  }
}

int main() {
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = 4;
  constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

  const int n = 1024 * 800;
  const int numTiles = (n + TILE_ITEMS - 1) / TILE_ITEMS;
  printf("numTiles: %d\n", numTiles);
  printf("tileItems: %d (block=%d, items/thread=%d)\n",
         TILE_ITEMS, BLOCK_THREADS, ITEMS_PER_THREAD);

  const int32_t size = n * sizeof(int);

  int* h_in = reinterpret_cast<int*>(malloc(size));
  int* h_out = reinterpret_cast<int*>(malloc(size));
  int* h_res = reinterpret_cast<int*>(malloc(size));

  for (int i = 0; i < n; ++i) {
    h_in[i] = i % 10;
  }
  prefix_sum_CPU<int>(h_in, h_res, n);

  int *d_in, *d_out, *d_state;
  int *d_tile_agg, *d_tile_prefix;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);
  cudaMalloc(&d_tile_agg, numTiles * sizeof(int));
  cudaMalloc(&d_tile_prefix, numTiles * sizeof(int));
  cudaMalloc(&d_state, numTiles * sizeof(int));

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  cudaMemset(d_state, 0, numTiles * sizeof(int));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int iters = 10000;

  for (int w = 0; w < 10; ++w) {
    prefix_sum_inclusive_dlb_optimized<int, BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<dim3(numTiles), dim3(BLOCK_THREADS)>>>(d_out, d_in, n,
                                                  d_tile_agg, d_tile_prefix,
                                                  d_state, w + 1);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; ++i) {
    prefix_sum_inclusive_dlb_optimized<int, BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<dim3(numTiles), dim3(BLOCK_THREADS)>>>(d_out, d_in, n,
                                                  d_tile_agg, d_tile_prefix,
                                                  d_state, i + 100);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  printf("total time (v4.4 single-kernel DLB optimized) = %.3f ms\n", ms);
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
  cudaFree(d_tile_agg);
  cudaFree(d_tile_prefix);
  cudaFree(d_state);

  return 0;
}
