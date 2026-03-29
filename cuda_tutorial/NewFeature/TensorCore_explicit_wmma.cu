#include <cmath>
#include <random>
#include <cstdlib>
#include <iostream>

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if __has_include(<cuda_fp4.h>)
#include <cuda_fp4.h>
#define HAS_CUDA_FP4 1
#else
#define HAS_CUDA_FP4 0
#endif

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
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;
constexpr int ELEMS_A = M * K;
constexpr int ELEMS_B = K * N;
constexpr int ELEMS_C = M * N;
}  // namespace

__global__ void wmma_gemm_16x16x16_kernel(const half* A,
                                          const half* B,
                                          float* C) {
  if (threadIdx.x >= warpSize) return;

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);
  wmma::load_matrix_sync(a_frag, A, K);
  wmma::load_matrix_sync(b_frag, B, N);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}

#if HAS_CUDA_FP4
__global__ void fp4_to_half_kernel(const __nv_fp4_e2m1* in, half* out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<half>(in[idx]);
  }
}
#endif

static void cpu_ref_gemm(const std::vector<float>& A,
                         const std::vector<float>& B,
                         std::vector<float>& C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

static float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
  float mx = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    mx = std::max(mx, std::fabs(a[i] - b[i]));
  }
  return mx;
}

int main() {
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name
            << " (SM " << prop.major << prop.minor << ")\n";

  if (prop.major < 7) {
    std::cout << "This demo needs Tensor Core capable GPU (SM70+).\n";
    return 0;
  }

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> hA_f32(ELEMS_A), hB_f32(ELEMS_B),
                     hC_ref(ELEMS_C), hC_tc(ELEMS_C);
  std::vector<half> hA_half(ELEMS_A), hB_half(ELEMS_B);

  for (int i = 0; i < ELEMS_A; ++i) {
    hA_f32[i] = dist(rng);
    hA_half[i] = __float2half(hA_f32[i]);
  }
  for (int i = 0; i < ELEMS_B; ++i) {
    hB_f32[i] = dist(rng);
    hB_half[i] = __float2half(hB_f32[i]);
  }

  cpu_ref_gemm(hA_f32, hB_f32, hC_ref);

  half* dA = nullptr;
  half* dB = nullptr;
  float* dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, ELEMS_A * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dB, ELEMS_B * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dC, ELEMS_C * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, hA_half.data(),
                        ELEMS_A * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB_half.data(),
                        ELEMS_B * sizeof(half), cudaMemcpyHostToDevice));

  wmma_gemm_16x16x16_kernel<<<1, 32>>>(dA, dB, dC);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(hC_tc.data(), dC,
                        ELEMS_C * sizeof(float), cudaMemcpyDeviceToHost));
  std::cout << "\n[WMMA FP16->FP32] explicit Tensor Core call done.\n";
  std::cout << "max |TC - CPU(ref)| = " << max_abs_diff(hC_tc, hC_ref) << "\n";

#if HAS_CUDA_FP4
  std::vector<__nv_fp4_e2m1> hA_fp4(ELEMS_A), hB_fp4(ELEMS_B);
  std::vector<half> hA_from_fp4(ELEMS_A), hB_from_fp4(ELEMS_B);
  std::vector<float> hA_fp4_ref(ELEMS_A),
                     hB_fp4_ref(ELEMS_B), hC_fp4_ref(ELEMS_C), hC_fp4_tc(ELEMS_C);

  for (int i = 0; i < ELEMS_A; ++i) {
    // quantize to FP4 (E2M1)
    hA_fp4[i] = __nv_fp4_e2m1(hA_f32[i]);

    // dequantize to half for WMMA path
    hA_from_fp4[i] = static_cast<half>(hA_fp4[i]);
    hA_fp4_ref[i] = __half2float(hA_from_fp4[i]);
  }
  for (int i = 0; i < ELEMS_B; ++i) {
    hB_fp4[i] = __nv_fp4_e2m1(hB_f32[i]);
    hB_from_fp4[i] = static_cast<half>(hB_fp4[i]);
    hB_fp4_ref[i] = __half2float(hB_from_fp4[i]);
  }
  cpu_ref_gemm(hA_fp4_ref, hB_fp4_ref, hC_fp4_ref);

  __nv_fp4_e2m1* dA_fp4 = nullptr;
  __nv_fp4_e2m1* dB_fp4 = nullptr;
  CHECK_CUDA(cudaMalloc(&dA_fp4, ELEMS_A * sizeof(__nv_fp4_e2m1)));
  CHECK_CUDA(cudaMalloc(&dB_fp4, ELEMS_B * sizeof(__nv_fp4_e2m1)));
  CHECK_CUDA(cudaMemcpy(dA_fp4, hA_fp4.data(),
                        ELEMS_A * sizeof(__nv_fp4_e2m1), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB_fp4, hB_fp4.data(),
                        ELEMS_B * sizeof(__nv_fp4_e2m1), cudaMemcpyHostToDevice));

  const int threads = 128;
  fp4_to_half_kernel<<<(ELEMS_A + threads - 1) / threads, threads>>>(
      dA_fp4, dA, ELEMS_A);
  fp4_to_half_kernel<<<(ELEMS_B + threads - 1) / threads, threads>>>(
      dB_fp4, dB, ELEMS_B);
  CHECK_CUDA(cudaGetLastError());

  wmma_gemm_16x16x16_kernel<<<1, 32>>>(dA, dB, dC);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(hC_fp4_tc.data(), dC,
                        ELEMS_C * sizeof(float), cudaMemcpyDeviceToHost));
  std::cout << "\n[FP4(E2M1) input + WMMA compute]\n";
  std::cout << "max |TC - CPU(ref from FP4)| = "
            << max_abs_diff(hC_fp4_tc, hC_fp4_ref) << "\n";
  std::cout <<
      "Note: Native FP4 Tensor Core MMA depends on newer arch/toolchain.\n";

  CHECK_CUDA(cudaFree(dA_fp4));
  CHECK_CUDA(cudaFree(dB_fp4));
#else
  std::cout << "\n cuda_fp4.h not found in this CUDA toolkit.\n";
#endif

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  return 0;
}
