#include <vector>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__)                \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                               	\
    }                                                                        	\
  } while (0)

namespace {
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_ELEMS = WMMA_M * WMMA_N;
}  // namespace

// ---------- SASS-leaning demos ----------
__global__ void ffma_kernel(const float* x, const float* y, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float a = x[i];
  float b = y[i];
  float acc = 0.0f;
#pragma unroll 16
  for (int k = 0; k < 64; ++k) {
    acc = fmaf(a, b, acc);
    a += 0.001f;
    b -= 0.001f;
  }
  out[i] = acc;
}

__global__ void hmma_wmma_fp16_kernel(const half* A, const half* B, float* C) {
  if (threadIdx.x >= warpSize) return;
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  wmma::load_matrix_sync(a_frag, A, WMMA_K);
  wmma::load_matrix_sync(b_frag, B, WMMA_N);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(C, c_frag, WMMA_N, wmma::mem_row_major);
}

// "DMMA representative": this is FP64 matmul workload.
// On architectures with FP64 Tensor Core path, SASS may show DMMA-like ops.
// On others, you will typically see DFMA.
__global__ void fp64_matmul_kernel(const double* A,
                                   const double* B,
                                   double* C,
                                   int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n || col >= n) return;
  double acc = 0.0;
#pragma unroll 8
  for (int k = 0; k < n; ++k) {
    acc = fma(A[row * n + k], B[k * n + col], acc);
  }
  C[row * n + col] = acc;
}

__global__ void imma_wmma_int8_kernel(const int8_t* A,
                                      const int8_t* B,
                                      int* C) {
  if (threadIdx.x >= warpSize) return;
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                 signed char, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                 signed char, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;
  wmma::fill_fragment(c_frag, 0);
  wmma::load_matrix_sync(a_frag, A, WMMA_K);
  wmma::load_matrix_sync(b_frag, B, WMMA_N);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(C, c_frag, WMMA_N, wmma::mem_row_major);
}

// ---------- PTX explicit demos ----------
__global__ void ptx_cp_async_mbarrier_redux_kernel(const int* in,
                                                   int* out_sum,
                                                   int* out_ready) {
  extern __shared__ int smem[];
  __shared__ unsigned long long bar;

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  if (tid == 0) {
    const uint32_t bar_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(bar_addr), "r"(blockDim.x) : "memory");
  }
  __syncthreads();

  const uint32_t saddr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&smem[tid]));
  const uint64_t gaddr =
      static_cast<uint64_t>(__cvta_generic_to_global(&in[gid]));
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %2;"
               :
               : "r"(saddr), "l"(gaddr), "n"(4)
               : "memory");
  asm volatile("cp.async.commit_group;" : : : "memory");
  asm volatile("cp.async.wait_group 0;" : : : "memory");
  __syncthreads();

  unsigned long long state = 0;
  int ready = 0;
  const uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
               : "=l"(state)
               : "r"(bar_addr)
               : "memory");
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "mbarrier.test_wait.shared.b64 p, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, p;\n\t"
               "}"
               : "=r"(ready)
               : "r"(bar_addr), "l"(state)
               : "memory");

  int sum = 0;
  const int v = smem[tid];
  asm volatile("redux.sync.add.s32 %0, %1, %2;"
               : "=r"(sum)
               : "r"(v), "r"(0xffffffff));

  if (tid == 0) {
    out_sum[0] = sum;
    out_ready[0] = ready;
  }
}

int main() {
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name
            << " (SM " << prop.major << prop.minor << ")\n";
  std::cout << "This demo maps code paths to representative SASS/PTX instructions.\n\n";

  // FFMA
  constexpr int n = 256;
  std::vector<float> hx(n, 1.25f), hy(n, 0.75f), hout(n, 0.0f);
  float *dx = nullptr, *dy = nullptr, *dout = nullptr;
  CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dout, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dx, hx.data(),
             n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dy, hy.data(),
             n * sizeof(float), cudaMemcpyHostToDevice));
  ffma_kernel<<<1, n>>>(dx, dy, dout);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(hout.data(), dout,
             n * sizeof(float), cudaMemcpyDeviceToHost));
  std::cout << "[FFMA] out[0] = " << hout[0] << "\n";

  // HMMA
  std::vector<half> hA(WMMA_ELEMS), hB(WMMA_ELEMS);
  std::vector<float> hC(WMMA_ELEMS, 0.0f);
  for (int i = 0; i < WMMA_ELEMS; ++i) {
    hA[i] = __float2half(0.5f);
    hB[i] = __float2half(1.0f);
  }
  half *dA_h = nullptr, *dB_h = nullptr;
  float* dC_hmma = nullptr;
  CHECK_CUDA(cudaMalloc(&dA_h, WMMA_ELEMS * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dB_h, WMMA_ELEMS * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dC_hmma, WMMA_ELEMS * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA_h, hA.data(),
                        WMMA_ELEMS * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB_h, hB.data(),
                        WMMA_ELEMS * sizeof(half), cudaMemcpyHostToDevice));
  hmma_wmma_fp16_kernel<<<1, 32>>>(dA_h, dB_h, dC_hmma);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(hC.data(), dC_hmma,
                        WMMA_ELEMS * sizeof(float), cudaMemcpyDeviceToHost));
  std::cout << "[HMMA] C[0] = " << hC[0] << "\n";

  // IMMA
  std::vector<int8_t> hAi8(WMMA_ELEMS, 1), hBi8(WMMA_ELEMS, 2);
  std::vector<int> hCi8(WMMA_ELEMS, 0);
  int8_t *dAi8 = nullptr, *dBi8 = nullptr;
  int* dCi8 = nullptr;
  CHECK_CUDA(cudaMalloc(&dAi8, WMMA_ELEMS * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&dBi8, WMMA_ELEMS * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&dCi8, WMMA_ELEMS * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dAi8, hAi8.data(),
                        WMMA_ELEMS * sizeof(int8_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dBi8, hBi8.data(),
                        WMMA_ELEMS * sizeof(int8_t), cudaMemcpyHostToDevice));
  imma_wmma_int8_kernel<<<1, 32>>>(dAi8, dBi8, dCi8);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(hCi8.data(), dCi8,
                        WMMA_ELEMS * sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "[IMMA] C[0] = " << hCi8[0] << "\n";

  // "DMMA representative" workload
  constexpr int dn = 16;
  std::vector<double> hAd(dn * dn, 1.0), hBd(dn * dn, 2.0), hCd(dn * dn, 0.0);
  double *dAd = nullptr, *dBd = nullptr, *dCd = nullptr;
  CHECK_CUDA(cudaMalloc(&dAd, hAd.size() * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&dBd, hBd.size() * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&dCd, hCd.size() * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(dAd, hAd.data(),
                        hAd.size() * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dBd, hBd.data(),
                        hBd.size() * sizeof(double), cudaMemcpyHostToDevice));
  fp64_matmul_kernel<<<dim3(1, 1), dim3(16, 16)>>>(dAd, dBd, dCd, dn);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(hCd.data(), dCd,
                        hCd.size() * sizeof(double), cudaMemcpyDeviceToHost));
  std::cout << "[DMMA-representative FP64] C[0] = " << hCd[0] << "\n";

  // PTX cp.async + mbarrier + redux.sync
  std::vector<int> hInWarp(32, 1);
  int *dInWarp = nullptr, *dSum = nullptr, *dReady = nullptr;
  int hSum = 0, hReady = 0;
  CHECK_CUDA(cudaMalloc(&dInWarp, 32 * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dSum, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dReady, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dInWarp, hInWarp.data(),
                        32 * sizeof(int), cudaMemcpyHostToDevice));
  ptx_cp_async_mbarrier_redux_kernel<<<1, 32, 32 * sizeof(int)>>>(
      dInWarp, dSum, dReady);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(&hSum, dSum, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&hReady, dReady, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "[PTX cp.async + mbarrier + redux.sync] sum = " << hSum
            << ", mbarrier_ready = " << hReady << "\n";

  std::cout << "\nInspect SASS/PTX with:\n";
  std::cout << "  cuobjdump --dump-sass ./build/NewFeature/Ampere_PTX_ISA_showcase\n";
  std::cout << "  cuobjdump --dump-ptx  ./build/NewFeature/Ampere_PTX_ISA_showcase\n";
  std::cout << "Look for tokens: FFMA / HMMA / IMMA / (DMMA or DFMA), cp.async, mbarrier, redux.sync.\n";

  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));
  CHECK_CUDA(cudaFree(dout));
  CHECK_CUDA(cudaFree(dA_h));
  CHECK_CUDA(cudaFree(dB_h));
  CHECK_CUDA(cudaFree(dC_hmma));
  CHECK_CUDA(cudaFree(dAi8));
  CHECK_CUDA(cudaFree(dBi8));
  CHECK_CUDA(cudaFree(dCi8));
  CHECK_CUDA(cudaFree(dAd));
  CHECK_CUDA(cudaFree(dBd));
  CHECK_CUDA(cudaFree(dCd));
  CHECK_CUDA(cudaFree(dInWarp));
  CHECK_CUDA(cudaFree(dSum));
  CHECK_CUDA(cudaFree(dReady));

  return 0;
}
