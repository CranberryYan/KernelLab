#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__)                \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                               	\
    }                                                                        	\
  } while (0)

__global__ void block_sum_sync(const float* in, float* out, int n) {
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;

  smem[tid] = (gid < n) ? in[gid] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) out[blockIdx.x] = smem[0];
}

__global__ void block_sum_async_g2s(const float* in, float* out, int n) {
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  cg::thread_block tb = cg::this_thread_block();

  #if 0
  if (num_per_sub > 0) {
    evs_in_out_src = tops::memcpy_async(
        ctx_in_out_src,
        tops::mdspan(tops::Global, output + sub_start, num_per_sub),
        tops::mdspan(tops::Global, input + sub_start, num_per_sub));
  }

  if (output != input && input_num > 0 && num_per_sub > 0) {
    evs_in_out_src.wait();
  }
  #endif

  if (gid < n) {
    // On SM80+, this maps to cp.async path.
    cg::memcpy_async(tb, &smem[tid], &in[gid], sizeof(float));
  } else {
    smem[tid] = 0.0f;
  }
  cg::wait(tb);
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) out[blockIdx.x] = smem[0];
}

template <typename KernelLauncher>
float launch_and_time_ms(KernelLauncher launch_once, int iters) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  launch_once();
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch_once();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return total_ms / static_cast<float>(iters);
}

int main() {
  constexpr int threads = 256;
  constexpr int blocks = 256;
  constexpr int n = threads * blocks;
  constexpr int iters = 500;

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::cout <<
      "GPU: " << prop.name << " (SM " << prop.major << prop.minor << ")\n";
  if (prop.major < 8) {
    std::cout << "Note: SM80+ has native cp.async acceleration.\n";
  }

  std::vector<float> hIn(n), hSync(blocks), hAsync(blocks);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < n; ++i) hIn[i] = dist(rng);

  float* dIn = nullptr;
  float* dSync = nullptr;
  float* dAsync = nullptr;
  CHECK_CUDA(cudaMalloc(&dIn, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dSync, blocks * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dAsync, blocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dIn, hIn.data(),
                        n * sizeof(float), cudaMemcpyHostToDevice));

  const size_t smem_bytes = static_cast<size_t>(threads) * sizeof(float);
  const auto launch_sync = [&]() {
    block_sum_sync<<<blocks, threads, smem_bytes>>>(dIn, dSync, n);
    CHECK_CUDA(cudaGetLastError());
  };
  const auto launch_async = [&]() {
    block_sum_async_g2s<<<blocks, threads, smem_bytes>>>(dIn, dAsync, n);
    CHECK_CUDA(cudaGetLastError());
  };

  const float sync_ms = launch_and_time_ms(launch_sync, iters);
  const float async_ms = launch_and_time_ms(launch_async, iters);

  CHECK_CUDA(cudaMemcpy(hSync.data(), dSync,
                        blocks * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hAsync.data(), dAsync,
                        blocks * sizeof(float), cudaMemcpyDeviceToHost));

  float max_diff = 0.0f;
  for (int i = 0; i < blocks; ++i) {
    max_diff = std::max(max_diff, std::fabs(hSync[i] - hAsync[i]));
  }

  std::cout << "\n[Sync copy]        " << sync_ms << " ms\n";
  std::cout << "[Async g->s copy]  " << async_ms << " ms\n";
  std::cout << "max |sync-async| = " << max_diff << "\n";

  CHECK_CUDA(cudaFree(dIn));
  CHECK_CUDA(cudaFree(dSync));
  CHECK_CUDA(cudaFree(dAsync));
  return 0;
}
