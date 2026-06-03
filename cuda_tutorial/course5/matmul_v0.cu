#include <cstdio>
#include <string>
#include <vector>
#include <random>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define TOL 1e-5f

void checkCudaError(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + " CUDA ERROR: " +
                             cudaGetErrorString(err));
  }
}
void checkCublasError(cublasStatus_t status, const char* msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(msg) + " CUBLAS ERROR: " +
                             std::to_string(status));
  }
}

// 内积(dot product):
//  a = [a1, a2, ..., ak]
//  b = [b1, b2, ..., bk]
//  a dot b = a1 * b1 + a2 * b2 + ... + ak * bk
//  a = [1, 2, 3]   b = [4, 5, 6]
//  a dot b = 4 + 10 + 18 = 32

// 外积(outer product)
//  第一个向量里的每个元素, 都去乘第二个向量里的所有元素, 最后得到一个矩阵
//  [1              [[4,   5,  6]
//   2  [4, 5, 6] =  [8,  10, 12]
//   3]              [12, 15, 18]]

// matmul
// A x B = C
// A: M x K
// B: K x N
// C: M x N
// 内积: C[m, n]: 沿着K维, 做内积(A的m行, B的n列的内积, loop MxN次(并行点))
// 外积: A按列拆分, B按行拆分, A的Ki列 outer B的Ki行, 累和K个矩阵

// 求一个元素, 最自然就是内积
// 高性能实现, 一般是外积, 因为拿一个tile

__global__ void mysgemm_v0(int M, int N, int K,
                           float alpha, float* A, float* B,
                           float beta, float* C) {
  int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (gid_x >= N || gid_y >= M) {
    return;
  }

  // 内积: C[m, n]: 沿着K维, 做内积(A的m行, B的n列的内积, loop MxN次(并行点))
  float tmp = 0.0f;
  for (int i = 0; i < K; ++i) {
    tmp += A[gid_y * K + i] * B[i * N + gid_x];
  }

  C[gid_y * N + gid_x] = alpha * tmp + beta * C[gid_y * N + gid_x];
}

int main() {
  std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

  std::ofstream csv_file("./course5/sgemm_benchmark_v0.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

  for (auto N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float* A = nullptr;
    float* B = nullptr;
    float* C_cublas = nullptr;
    float* C_v0 = nullptr;
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_v0 = nullptr;
    cublasHandle_t handle = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    bool out_of_mem = false;

    try {
      A = reinterpret_cast<float*>(malloc(size));
      B = reinterpret_cast<float*>(malloc(size));
      C_cublas = reinterpret_cast<float*>(malloc(size));
      C_v0 = reinterpret_cast<float*>(malloc(size));
      if (A == nullptr || B == nullptr ||
          C_cublas == nullptr || C_v0 == nullptr) {
        throw std::runtime_error("Host malloc failed");
      }

      checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
      checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
      checkCudaError(cudaMalloc(&d_C_v0, size), "cudaMalloc d_C_v0 failed");

      float alpha = 2.0f;
      float beta = 0.0f;
      std::mt19937 rng(12345);  // 固定 seed，便于复现
      std::uniform_real_distribution<float> dist(-10.0f, 10.0f);      
      for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
      }

      checkCudaError(cudaMemset(d_C_v0, 0, size), "cudaMemset d_C_v0 failed");

      checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy A to device failed");

      checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy B to device failed");

      checkCublasError(cublasCreate(&handle), "cublasCreate failed");

      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

      // warm up
      int warmup_time = 10;
      for (int i = 0; i < warmup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v0, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v0, N),
                         "cublasSgemm failed");
      }

      checkCudaError(cudaEventRecord(stop),
                     "cudaEventRecord(stop cublas) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize cublas failed");

      float cublas_time = 0;
      checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                     "cudaEventElapsedTime cublas failed");

      checkCudaError(cudaMemcpy(C_cublas, d_C_v0, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v0
      checkCudaError(cudaMemset(d_C_v0, 0, size), "cudaMemset d_C_v0 failed");

      int BLOCK_SIZE_X = BLOCK_SIZE;
      int BLOCK_SIZE_Y = BLOCK_SIZE;
      dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
      int GRID_SIZE_X = (N + threads.x - 1) / threads.x;
      int GRID_SIZE_Y = (N + threads.y - 1) / threads.y;
      dim3 blocks(GRID_SIZE_X, GRID_SIZE_Y);

      for (int i = 0; i < warmup_time; ++i) {
        mysgemm_v0<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v0);
        checkCudaError(cudaGetLastError(), "mysgemm_v0 warmup launch failed");
      }
      checkCudaError(cudaDeviceSynchronize(), "mysgemm_v0 warmup sync failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v0) failed");
      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v0<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v0);
        checkCudaError(cudaGetLastError(), "mysgemm_v0 launch failed");
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v0) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v0 failed");

      float v0_time = 0;
      checkCudaError(cudaEventElapsedTime(&v0_time, start, stop),
                     "cudaEventElapsedTime v0 failed");

      checkCudaError(cudaMemcpy(C_v0, d_C_v0, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v0 failed");

      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        float ref  = C_cublas[i];
        float got  = C_v0[i];
        float diff = fabsf(ref - got);
        float tol  = 1e-3f + 1e-3f * fabsf(ref);

        if (i < 10) {
          printf("ref: %f, got: %f\n", ref, got);
        }

        if (diff > tol) {
          ++error_count;
          printf("idx=%d ref=%f got=%f diff=%f tol=%f\n",
                  i, ref, got, diff, tol);
        }
      }

      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
      float v0_gflops =
          repeat_time * 2.0f * N * N * N / (v0_time * 1e6f);  // GFlops

      printf("error_count: %d\n", error_count);
      csv_file << N << "," << cublas_gflops << "," << v0_gflops << ","
               << (error_count == 0 ? "1" : "0") << std::endl;

    } catch (const std::exception& e) {
      std::cerr << "Error during testing size " << N << ": " << e.what()
                << std::endl;
      out_of_mem = true;
    }

    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    if (handle != nullptr) {
      cublasDestroy(handle);
    }
    if (d_A != nullptr) {
      cudaFree(d_A);
    }
    if (d_B != nullptr) {
      cudaFree(d_B);
    }
    if (d_C_v0 != nullptr) {
      cudaFree(d_C_v0);
    }
    if (A != nullptr) {
      free(A);
    }
    if (B != nullptr) {
      free(B);
    }
    if (C_cublas != nullptr) {
      free(C_cublas);
    }
    if (C_v0 != nullptr) {
      free(C_v0);
    }

    if (!out_of_mem) {
      std::cout << "Finished size: " << N << std::endl;
    } else {
      csv_file << N << ",OOM,OOM,0" << std::endl;
    }
  }

  csv_file.close();

  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v0.csv'"
            << std::endl;


  return 0;
}
