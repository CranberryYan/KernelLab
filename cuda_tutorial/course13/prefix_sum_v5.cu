#include <cstdio>
#include <cstdint>
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

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      printf("CUDA error at %s:%d -> %s\n", __FILE__, __LINE__,              \
             cudaGetErrorString(err__));                                      \
      return false;                                                           \
    }                                                                         \
  } while (0)

struct BenchmarkCase {
  int n;
  int iters;
  int warmup;
  const char* name;
};

static int next_pow2(int x) {
  int p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

static bool verify_result_sampled(const int* out, const int* ref, int n) {
  if (n <= 0) {
    return true;
  }
  const int fixed_idx[] = {0, 1, 2, 3, 7, 15, 31, 63};
  for (int idx : fixed_idx) {
    if (idx < n && out[idx] != ref[idx]) {
      printf("verify failed @%d: out=%d ref=%d\n", idx, out[idx], ref[idx]);
      return false;
    }
  }
  for (int k = 1; k <= 8; ++k) {
    int idx = (int)(((int64_t)n * k) / 9);
    if (idx >= n) {
      idx = n - 1;
    }
    if (out[idx] != ref[idx]) {
      printf("verify failed @%d: out=%d ref=%d\n", idx, out[idx], ref[idx]);
      return false;
    }
  }
  if (out[n - 1] != ref[n - 1]) {
    printf("verify failed @%d: out=%d ref=%d\n", n - 1, out[n - 1], ref[n - 1]);
    return false;
  }
  return true;
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

static bool run_benchmark_case(const BenchmarkCase& bc) {
  const int thread_per_block = 1024;
  int logical_blocks = (bc.n + thread_per_block - 1) / thread_per_block;
  int numBlocks = next_pow2(logical_blocks);
  if (numBlocks > thread_per_block) {
    printf("[%-12s] n=%-10d logical=%-7d padded=%-7d  SKIP (K2 supports <=1024)\n",
           bc.name, bc.n, logical_blocks, numBlocks);
    return true;
  }

  const int32_t size = bc.n * (int)sizeof(int);
  const int32_t smem_blocks = thread_per_block * (int)sizeof(int);
  const int32_t smem_sums = thread_per_block * (int)sizeof(int);

  int* h_in = reinterpret_cast<int*>(malloc(size));
  int* h_out = reinterpret_cast<int*>(malloc(size));
  int* h_ref = reinterpret_cast<int*>(malloc(size));
  if (!h_in || !h_out || !h_ref) {
    printf("host malloc failed\n");
    free(h_in);
    free(h_out);
    free(h_ref);
    return false;
  }

  for (int i = 0; i < bc.n; ++i) {
    h_in[i] = i % 10;
  }
  prefix_sum_CPU<int>(h_in, h_ref, bc.n);

  int *d_in, *d_out, *d_sums, *d_partial;
  CUDA_CHECK(cudaMalloc(&d_in, size));
  CUDA_CHECK(cudaMalloc(&d_out, size));
  CUDA_CHECK(cudaMalloc(&d_sums, numBlocks * (int)sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_partial, size));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int w = 0; w < bc.warmup; ++w) {
    scan_blocks<int><<<dim3(numBlocks), dim3(thread_per_block), smem_blocks>>>(
        d_partial, d_sums, d_in, bc.n);
    scan_block_sums<int><<<dim3(1), dim3(thread_per_block), smem_sums>>>(
        d_sums, numBlocks);
    add_block_prefix<int><<<dim3(numBlocks), dim3(thread_per_block)>>>(
        d_out, d_partial, d_sums, bc.n);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start, 0));
  for (int i = 0; i < bc.iters; ++i) {
    scan_blocks<int><<<dim3(numBlocks), dim3(thread_per_block), smem_blocks>>>(
        d_partial, d_sums, d_in, bc.n);
    scan_block_sums<int><<<dim3(1), dim3(thread_per_block), smem_sums>>>(
        d_sums, numBlocks);
    add_block_prefix<int><<<dim3(numBlocks), dim3(thread_per_block)>>>(
        d_out, d_partial, d_sums, bc.n);
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / bc.iters;
  const double elems_per_s = (double)bc.n / (avg_ms * 1e-3);
  // v5 has an extra full-size buffer round-trip (d_partial write+read)
  const double est_bytes = (double)bc.n * sizeof(int) * 4.0;
  const double gib_per_s =
      est_bytes / (avg_ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);

  CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
  const bool ok = verify_result_sampled(h_out, h_ref, bc.n);

  printf("[%-12s] n=%-10d logical=%-7d padded=%-7d avg=%.6f ms  "
         "throughput=%.3f Gelem/s  est_bw=%.3f GiB/s  %s\n",
         bc.name, bc.n, logical_blocks, numBlocks, avg_ms, elems_per_s / 1e9,
         gib_per_s, ok ? "OK" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaFree(d_partial));
  free(h_in);
  free(h_out);
  free(h_ref);
  return ok;
}

int main() {
  printf("v5 benchmark (three-kernel baseline)\n");
  printf("config: block=1024, K2 single-block scan on padded block sums\n");

  const BenchmarkCase cases[] = {
      {1024 * 64,   20000, 20, "small-64k"},
      {1024 * 800,  10000, 10, "mid-800k"},
      {1024 * 1024, 5000,  10, "max-1m"},
      {1024 * 4096, 2000,  10, "large-4m"},
  };

  bool all_ok = true;
  for (const auto& bc : cases) {
    bool ok = run_benchmark_case(bc);
    all_ok = all_ok && ok;
  }

  printf("benchmark done: %s\n", all_ok ? "ALL PASS" : "HAS FAILURES");
  return all_ok ? 0 : 1;
}
