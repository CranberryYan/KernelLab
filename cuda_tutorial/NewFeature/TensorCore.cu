#include <random>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__)                \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                               	\
    }                                                                        	\
  } while (0)

#define CHECK_CUBLAS(call)                                                  	\
  do {                                                                      	\
    cublasStatus_t st__ = (call);                                            	\
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                    	\
      std::cerr << "cuBLAS error code: " << static_cast<int>(st__)          	\
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                               	\
    }                                                                        	\
  } while (0)

static float max_abs_diff(const std::vector<float>& a,
													const std::vector<float>& b) {
	float max_d = 0.0f;
	for (size_t i = 0; i < a.size(); ++i) {
		max_d = std::max(max_d, std::fabs(a[i] - b[i]));
	}
	return max_d;
}
static double max_abs_diff(const std::vector<double>& a,
													 const std::vector<double>& b) {
	double max_d = 0.0;
	for (size_t i = 0; i < a.size(); ++i) {
		max_d = std::max(max_d, std::fabs(a[i] - b[i]));
	}
	return max_d;
}
static double tflops_from_ms(int n, float ms) {
	const double ops = 2.0 * static_cast<double>(n) * n * n;
	return ops / (static_cast<double>(ms) * 1.0e-3) / 1.0e12;
}

static float time_gemm_ex(cublasHandle_t handle,
													cublasComputeType_t compute_type,
													cublasGemmAlgo_t algo,
													int n,
													const void* A, cudaDataType_t Atype,
												  const void* B, cudaDataType_t Btype,
												  void* C, cudaDataType_t Ctype,
													const void* alpha, const void* beta, int iters) {
	cudaEvent_t start, stop;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	// warm up
	CHECK_CUBLAS(cublasGemmEx(handle,
														CUBLAS_OP_N, CUBLAS_OP_N,
													 	n, n, n,
														alpha,
														B, Btype, n,
														A, Atype, n,
														beta,
														C, Ctype, n,
														compute_type,
														algo));

	CHECK_CUDA(cudaEventRecord(start));
	for (int i = 0; i < iters; ++i) {
		CHECK_CUBLAS(cublasGemmEx(handle,
															CUBLAS_OP_N, CUBLAS_OP_N,
															n, n, n,
															alpha,
															B, Btype, n,
															A, Atype, n,
															beta,
															C, Ctype, n,
															compute_type,
															algo));
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
	int dev = 0;
	cudaDeviceProp prop{};
	CHECK_CUDA(cudaGetDevice(&dev));
	CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

	std::cout << "GPU: " << prop.name << " (SM " << prop.major << prop.minor << ")\n";
  if (prop.major < 8) {
    std::cout << "This demo targets Ampere+ (SM80/86+). Skip.\n";
    return 0;
  }

  constexpr int n = 512;
  constexpr int iters = 50;
  const size_t elems = static_cast<size_t>(n) * n;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> hAf(elems), hBf(elems), hC_fast(elems), hC_ref(elems);
  std::vector<__nv_bfloat16> hAbf16(elems), hBbf16(elems);
  std::vector<double> hAd(elems), hBd(elems), hCd_fast(elems), hCd_ref(elems);

  for (size_t i = 0; i < elems; ++i) {
    hAf[i] = dist(rng);
    hBf[i] = dist(rng);
    hAbf16[i] = __float2bfloat16(hAf[i]);
    hBbf16[i] = __float2bfloat16(hBf[i]);
    hAd[i] = static_cast<double>(hAf[i]);
    hBd[i] = static_cast<double>(hBf[i]);
  }

  float *dAf = nullptr, *dBf = nullptr, *dCfast = nullptr, *dCref = nullptr;
  __nv_bfloat16 *dAbf16 = nullptr, *dBbf16 = nullptr;
  double *dAd = nullptr, *dBd = nullptr, *dCdfast = nullptr, *dCdref = nullptr;

  CHECK_CUDA(cudaMalloc(&dAf, elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dBf, elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dCfast, elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dCref, elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dAbf16, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dBbf16, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dAd, elems * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&dBd, elems * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&dCdfast, elems * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&dCdref, elems * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(dAf, hAf.data(),
												elems * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dBf, hBf.data(),
												elems * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dAbf16, hAbf16.data(),
												elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dBbf16, hBbf16.data(),
												elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dAd, hAd.data(),
												elems * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dBd, hBd.data(),
												elems * sizeof(double), cudaMemcpyHostToDevice));

	cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));

  {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float ms_fast = time_gemm_ex(handle,
                                       CUBLAS_COMPUTE_32F_FAST_TF32,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                                       n,
                                       dAf, CUDA_R_32F,
                                       dBf, CUDA_R_32F,
                                       dCfast, CUDA_R_32F,
                                       &alpha, &beta, iters);
    const float ms_ref = time_gemm_ex(handle,
                                      CUBLAS_COMPUTE_32F_PEDANTIC,
                                      CUBLAS_GEMM_DEFAULT,
                                      n,
                                      dAf, CUDA_R_32F,
                                      dBf, CUDA_R_32F,
                                      dCref, CUDA_R_32F,
                                      &alpha, &beta, iters);
    CHECK_CUDA(cudaMemcpy(hC_fast.data(), dCfast,
													elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_ref.data(), dCref,
													elems * sizeof(float), cudaMemcpyDeviceToHost));
    const float diff = max_abs_diff(hC_fast, hC_ref);
    std::cout << "\n[TF32] float input + Tensor Core fast TF32\n";
    std::cout << "  fast : " << ms_fast << " ms, "
							<< tflops_from_ms(n, ms_fast) << " TFLOPS\n";
    std::cout << "  ref  : " << ms_ref << " ms, "
							<< tflops_from_ms(n, ms_ref) << " TFLOPS\n";
    std::cout << "  max |fast-ref| = " << diff << "\n";
  }

  {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float ms_fast = time_gemm_ex(handle,
                                       CUBLAS_COMPUTE_32F_FAST_16BF,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                                       n,
                                       dAbf16, CUDA_R_16BF,
                                       dBbf16, CUDA_R_16BF,
                                       dCfast, CUDA_R_32F,
                                       &alpha, &beta, iters);
    const float ms_ref = time_gemm_ex(handle,
                                      CUBLAS_COMPUTE_32F_PEDANTIC,
                                      CUBLAS_GEMM_DEFAULT,
                                      n,
                                      dAbf16, CUDA_R_16BF,
                                      dBbf16, CUDA_R_16BF,
                                      dCref, CUDA_R_32F,
                                      &alpha, &beta, iters);
    CHECK_CUDA(cudaMemcpy(hC_fast.data(), dCfast,
													elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_ref.data(), dCref,
													elems * sizeof(float), cudaMemcpyDeviceToHost));
    const float diff = max_abs_diff(hC_fast, hC_ref);
    std::cout << "\n[BF16] bf16 input + fp32 accumulate\n";
    std::cout << "  fast : " << ms_fast << " ms, "
							<< tflops_from_ms(n, ms_fast) << " TFLOPS\n";
    std::cout << "  ref  : " << ms_ref << " ms, "
							<< tflops_from_ms(n, ms_ref) << " TFLOPS\n";
    std::cout << "  max |fast-ref| = " << diff << "\n";
  }

  {
    const double alpha = 1.0;
    const double beta = 0.0;
    const float ms_fast = time_gemm_ex(handle,
                                       CUBLAS_COMPUTE_64F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                                       n,
                                       dAd, CUDA_R_64F,
                                       dBd, CUDA_R_64F,
                                       dCdfast, CUDA_R_64F,
                                       &alpha, &beta, iters);
    const float ms_ref = time_gemm_ex(handle,
                                      CUBLAS_COMPUTE_64F_PEDANTIC,
                                      CUBLAS_GEMM_DEFAULT,
                                      n,
                                      dAd, CUDA_R_64F,
                                      dBd, CUDA_R_64F,
                                      dCdref, CUDA_R_64F,
                                      &alpha, &beta, iters);
    CHECK_CUDA(cudaMemcpy(hCd_fast.data(), dCdfast,
													elems * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hCd_ref.data(), dCdref,
													elems * sizeof(double), cudaMemcpyDeviceToHost));
    const double diff = max_abs_diff(hCd_fast, hCd_ref);
    std::cout << "\n[FP64] double input + double accumulate\n";
    std::cout << "  fast : " << ms_fast << " ms, "
							<< tflops_from_ms(n, ms_fast) << " TFLOPS\n";
    std::cout << "  ref  : " << ms_ref << " ms, "
							<< tflops_from_ms(n, ms_ref) << " TFLOPS\n";
    std::cout << "  max |fast-ref| = " << diff << "\n";
    std::cout <<
				"  FP64 Tensor Core acceleration is strongest on A100-class GPUs.\n";
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(dAf));
  CHECK_CUDA(cudaFree(dBf));
  CHECK_CUDA(cudaFree(dCfast));
  CHECK_CUDA(cudaFree(dCref));
  CHECK_CUDA(cudaFree(dAbf16));
  CHECK_CUDA(cudaFree(dBbf16));
  CHECK_CUDA(cudaFree(dAd));
  CHECK_CUDA(cudaFree(dBd));
  CHECK_CUDA(cudaFree(dCdfast));
  CHECK_CUDA(cudaFree(dCdref));


	return 0;
}
