// 实验1: 测似API

// ==== Benchmark ====
// sync_copy  time: 0.651432 ms, effective BW: 824.14 GB/s
// async_copy time: 0.649844 ms, effective BW: 826.15 GB/s

// 没多大差距, 是正常的

#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
      cudaError_t err = (call);                                               \
      if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err));                 \
        std::exit(EXIT_FAILURE);                                              \
      }                                                                       \
  } while (0)

template<int BLOCK_SIZE>
__global__ void sync_copy_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * BLOCK_SIZE + tid;

  __shared__ float smem[BLOCK_SIZE];

  if (gid < N) {
    smem[tid] = in[gid];
  }

  __syncthreads();

  if (gid < N) {
    out[gid] = smem[tid];
  }
}

template<int BLOCK_SIZE>
__global__ void async_copy_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * BLOCK_SIZE + tid;

  __shared__ float smem[BLOCK_SIZE];

  if (gid < N) {
    __pipeline_memcpy_async(&smem[tid], &in[gid], sizeof(float));
  }

  // 提交当前线程发起的 async copy group
  __pipeline_commit();

  // 等待之前提交的 async copy 全部完成
  __pipeline_wait_prior(0);

  // 这里保留 __syncthreads：
  // 当前例子每个线程只读自己搬的数据，理论上不一定需要；
  // 但作为 gmem -> smem -> gmem 的通用模板，保留它更符合后续 tile 共享场景。
  __syncthreads();

  if (gid < N) {
    out[gid] = smem[tid];
  }
}

float benchmark_kernel(void (*launcher)(const float*, float*, int),
                       const float* d_in,
                       float* d_out,
                       int N,
                       int warmup,
                       int repeat) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
    launcher(d_in, d_out, N);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    launcher(d_in, d_out, N);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return ms / repeat;
}

template<int BLOCK_SIZE>
void launch_sync_copy(const float* d_in, float* d_out, int N) {
  int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sync_copy_kernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_in, d_out, N);
}

template<int BLOCK_SIZE>
void launch_async_copy(const float* d_in, float* d_out, int N) {
  int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  async_copy_kernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_in, d_out, N);
}

bool check_result(const std::vector<float>& ref,
                const std::vector<float>& got,
                int N) {
  for (int i = 0; i < N; ++i) {
      float a = ref[i];
      float b = got[i];
      if (std::fabs(a - b) > 1e-6f) {
        printf("Mismatch at %d: ref = %.8f, got = %.8f\n", i, a, b);
        return false;
      }
  }
  return true;
}

int main() {
  constexpr int BLOCK_SIZE = 256;

  // N 取大一点，避免 launch overhead 占比过高。
  // 这里是 64M floats = 256 MB input。
  int N = 64 * 1024 * 1024;

  size_t bytes = static_cast<size_t>(N) * sizeof(float);

  printf("N               = %d\n", N);
  printf("Input bytes     = %.2f MB\n", bytes / 1024.0 / 1024.0);
  printf("BLOCK_SIZE      = %d\n", BLOCK_SIZE);

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  printf("GPU             = %s\n", prop.name);
  printf("Compute capability = %d.%d\n", prop.major, prop.minor);

  std::vector<float> h_in(N);
  std::vector<float> h_sync(N, 0.0f);
  std::vector<float> h_async(N, 0.0f);

  for (int i = 0; i < N; ++i) {
    h_in[i] = static_cast<float>(i % 1000) * 0.001f;
  }

  float* d_in = nullptr;
  float* d_sync = nullptr;
  float* d_async = nullptr;

  CHECK_CUDA(cudaMalloc(&d_in, bytes));
  CHECK_CUDA(cudaMalloc(&d_sync, bytes));
  CHECK_CUDA(cudaMalloc(&d_async, bytes));

  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_sync, 0, bytes));
  CHECK_CUDA(cudaMemset(d_async, 0, bytes));

  launch_sync_copy<BLOCK_SIZE>(d_in, d_sync, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  launch_async_copy<BLOCK_SIZE>(d_in, d_async, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_sync.data(), d_sync, bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_async.data(), d_async, bytes, cudaMemcpyDeviceToHost));

  bool sync_ok = check_result(h_in, h_sync, N);
  bool async_ok = check_result(h_in, h_async, N);

  printf("sync_copy result  : %s\n", sync_ok ? "PASS" : "FAIL");
  printf("async_copy result : %s\n", async_ok ? "PASS" : "FAIL");

  constexpr int WARMUP = 10;
  constexpr int REPEAT = 100;

  float sync_ms = benchmark_kernel(
      launch_sync_copy<BLOCK_SIZE>, d_in, d_sync, N, WARMUP, REPEAT);

  float async_ms = benchmark_kernel(
      launch_async_copy<BLOCK_SIZE>, d_in, d_async, N, WARMUP, REPEAT);

  // 当前 kernel 实际有一次 global read + 一次 global write。
  // gmem -> smem 是 read，smem -> gmem 是 write。
  double total_bytes = static_cast<double>(bytes) * 2.0;

  double sync_bw = total_bytes / (sync_ms * 1e-3) / 1e9;
  double async_bw = total_bytes / (async_ms * 1e-3) / 1e9;

  printf("\n==== Benchmark ====\n");
  printf("sync_copy  time: %.6f ms, effective BW: %.2f GB/s\n",
          sync_ms, sync_bw);
  printf("async_copy time: %.6f ms, effective BW: %.2f GB/s\n",
          async_ms, async_bw);

  printf("\nNote:\n");
  printf("This experiment copies global -> shared -> global.\n");
  printf("cp.async may not be faster here because we immediately wait after copy.\n");
  printf("The purpose is to verify API semantics, not yet latency hiding.\n");

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_sync));
  CHECK_CUDA(cudaFree(d_async));

  return 0;
}