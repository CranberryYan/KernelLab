// Exp3: cp.async copy 粒度 4B / 8B / 16B

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

template<typename VecT, int BLOCK_SIZE>
__global__ void sync_vec_copy_kernel(const VecT* __restrict__ in,
                                     VecT* __restrict__ out,
                                     int Nvec) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * BLOCK_SIZE + tid;

  __shared__ VecT smem[BLOCK_SIZE];

  if (gid < Nvec) {
    smem[tid] = in[gid];
  }

  __syncthreads();

  if (gid < Nvec) {
    out[gid] = smem[tid];
  }
}

template<typename VecT, int BLOCK_SIZE>
__global__ void async_vec_copy_kernel(const VecT* __restrict__ in,
                                      VecT* __restrict__ out,
                                      int Nvec) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * BLOCK_SIZE + tid;

  __shared__ VecT smem[BLOCK_SIZE];

  if (gid < Nvec) {
    __pipeline_memcpy_async(&smem[tid], &in[gid], sizeof(Nvec));
  }

  // 提交当前线程发起的 async copy group
  __pipeline_commit();

  // 等待之前提交的 async copy 全部完成
  __pipeline_wait_prior(0);

  // 这里保留 __syncthreads：
  // 当前例子每个线程只读自己搬的数据，理论上不一定需要；
  // 但作为 gmem -> smem -> gmem 的通用模板，保留它更符合后续 tile 共享场景。
  __syncthreads();

  if (gid < Nvec) {
    out[gid] = smem[tid];
  }
}

template<typename Launcher>
float benchmark(Launcher launcher, int warmup, int repeat) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
    launcher();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    launcher();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return ms / repeat;
}

template<typename VecT, int BLOCK_SIZE>
void launch_sync_vec_copy(const VecT* d_in, VecT* d_out, int Nvec) {
  int grid = (Nvec + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sync_vec_copy_kernel<VecT, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_in, d_out, Nvec);
}

template<typename VecT, int BLOCK_SIZE>
void launch_async_vec_copy(const VecT* d_in, VecT* d_out, int Nvec) {
  int grid = (Nvec + BLOCK_SIZE - 1) / BLOCK_SIZE;
  async_vec_copy_kernel<VecT, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_in, d_out, Nvec);
}

template<typename VecT>
void fill_input(std::vector<VecT>& h) {
  unsigned char* p = reinterpret_cast<unsigned char*>(h.data());
  size_t bytes = h.size() * sizeof(VecT);

  for (size_t i = 0; i < bytes; ++i) {
    p[i] = static_cast<unsigned char>((i * 131 + 17) & 0xff);
  }
}

template<typename VecT>
bool check_bytes(const std::vector<VecT>& ref,
                 const std::vector<VecT>& got,
                 int Nvec,
                 const char* name) {
  size_t bytes = static_cast<size_t>(Nvec) * sizeof(VecT);

  const unsigned char* a = reinterpret_cast<const unsigned char*>(ref.data());
  const unsigned char* b = reinterpret_cast<const unsigned char*>(got.data());

  for (size_t i = 0; i < bytes; ++i) {
    if (a[i] != b[i]) {
      printf("%s mismatch at byte %zu: expected = %u, got = %u\n",
              name,
              i,
              static_cast<unsigned>(a[i]),
              static_cast<unsigned>(b[i]));
      return false;
    }
  }

  return true;
}

template<typename VecT, int BLOCK_SIZE>
void run_case(const char* name,
              size_t total_bytes,
              int warmup,
              int repeat) {
  int Nvec = static_cast<int>(total_bytes / sizeof(VecT));
  size_t bytes = static_cast<size_t>(Nvec) * sizeof(VecT);

  std::vector<VecT> h_in(Nvec);
  std::vector<VecT> h_sync(Nvec);
  std::vector<VecT> h_async(Nvec);

  fill_input(h_in);

  VecT* d_in = nullptr;
  VecT* d_sync = nullptr;
  VecT* d_async = nullptr;

  CHECK_CUDA(cudaMalloc(&d_in, bytes));
  CHECK_CUDA(cudaMalloc(&d_sync, bytes));
  CHECK_CUDA(cudaMalloc(&d_async, bytes));

  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_sync, 0, bytes));
  CHECK_CUDA(cudaMemset(d_async, 0, bytes));

  launch_sync_vec_copy<VecT, BLOCK_SIZE>(d_in, d_sync, Nvec);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  launch_async_vec_copy<VecT, BLOCK_SIZE>(d_in, d_async, Nvec);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_sync.data(), d_sync, bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_async.data(), d_async, bytes, cudaMemcpyDeviceToHost));

  bool sync_ok = check_bytes(h_in, h_sync, Nvec, "sync");
  bool async_ok = check_bytes(h_in, h_async, Nvec, "async");

  auto sync_launcher = [&]() {
      launch_sync_vec_copy<VecT, BLOCK_SIZE>(d_in, d_sync, Nvec);
  };

  auto async_launcher = [&]() {
      launch_async_vec_copy<VecT, BLOCK_SIZE>(d_in, d_async, Nvec);
  };

  float sync_ms = benchmark(sync_launcher, warmup, repeat);
  float async_ms = benchmark(async_launcher, warmup, repeat);

  // global read + global write
  double traffic_bytes = static_cast<double>(bytes) * 2.0;

  double sync_bw = traffic_bytes / (sync_ms * 1e-3) / 1e9;
  double async_bw = traffic_bytes / (async_ms * 1e-3) / 1e9;

  int bytes_per_thread = static_cast<int>(sizeof(VecT));
  int bytes_per_warp = bytes_per_thread * 32;

  printf("==== %s ====\n", name);
  printf("sizeof(VecT)          = %d B\n", bytes_per_thread);
  printf("bytes per warp copy   = %d B\n", bytes_per_warp);
  printf("Nvec                  = %d\n", Nvec);
  printf("total payload         = %.2f MB\n", bytes / 1024.0 / 1024.0);
  printf("sync result           = %s\n", sync_ok ? "PASS" : "FAIL");
  printf("async result          = %s\n", async_ok ? "PASS" : "FAIL");
  printf("sync_copy  time       = %.6f ms, effective BW = %.2f GB/s\n",
          sync_ms, sync_bw);
  printf("async_copy time       = %.6f ms, effective BW = %.2f GB/s\n",
          async_ms, async_bw);
  printf("\n");

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_sync));
  CHECK_CUDA(cudaFree(d_async));
}

int main() {
  constexpr int BLOCK_SIZE = 256;

  // 固定总 payload 为 256 MB。
  // float  : 64M elements
  // float2 : 32M elements
  // float4 : 16M elements
  size_t total_bytes = 256ull * 1024ull * 1024ull;

  constexpr int WARMUP = 10;
  constexpr int REPEAT = 100;

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  printf("GPU                = %s\n", prop.name);
  printf("Compute capability = %d.%d\n", prop.major, prop.minor);
  printf("BLOCK_SIZE         = %d\n", BLOCK_SIZE);
  printf("Total payload      = %.2f MB\n\n", total_bytes / 1024.0 / 1024.0);

  run_case<float,  BLOCK_SIZE>("4B  cp.async copy: float",  total_bytes, WARMUP, REPEAT);
  run_case<float2, BLOCK_SIZE>("8B  cp.async copy: float2", total_bytes, WARMUP, REPEAT);
  run_case<float4, BLOCK_SIZE>("16B cp.async copy: float4", total_bytes, WARMUP, REPEAT);

  printf("Interpretation:\n");
  printf("4B  : each warp issues 32 * 4B  = 128B copy payload.\n");
  printf("8B  : each warp issues 32 * 8B  = 256B copy payload.\n");
  printf("16B : each warp issues 32 * 16B = 512B copy payload.\n");
  printf("This experiment still waits immediately after copy, so it mainly tests copy granularity, not latency hiding.\n");

  return 0;
}