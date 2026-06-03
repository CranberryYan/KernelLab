#include <cstdio>
#include <random>
#include <vector>
#include <ostream>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define TOL 1e-5f

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

// BLOCK_SIZE: 32
// dim3 blockDim(1024);
// dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32));

// bid_y -> A_tile, bid_x -> B_tile
// tile大小为[32, 32], 所以每个block中有1024个thread,
// 按照tid_x tid_y, 求当前tile的scale output
template <const int BLOCK_SIZE>
__global__ void mysgemm_v1(int M, int N, int K,
                           float alpha, float* A, float* B,
                           float beta, float* C) {
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  // 一个block要处理的tile尺寸(tile: 32 x 32, 1024thread_per_block)
  // Ctile[BM x BN] = Atile[BM x BK] x Btile[BK x BN]
  // 当前这个block负责
  //  Ctile[bid_y * BM : (bid_y+1) * BM, bid_x * BN : (bid_x+1) * BN]
  const int BM = BLOCK_SIZE;
  const int BN = BLOCK_SIZE;
  const int BK = BLOCK_SIZE;

  int tid_x = threadIdx.x % BK;
  int tid_y = threadIdx.x / BK;

  __shared__ float Asmem[BM * BK];
  __shared__ float Bsmem[BK * BN];

  // A([M, K]): 行之间是独立的, A_tile: y方向取tile([BM, BK]), stride: K
  //  loop方向: x轴, loop_stride: BK
  // B([K, N]): 列之间是独立的, B_tile: x方向取tile([BK, BN]), stride: 1
  //  loop方向: y轴, loop_stride: BK * N
  A = &A[bid_y * BM * K];
  B = &B[bid_x * BN];
  C = &C[bid_y * BM * N + bid_x * BN];

  float tmp = 0.0f;

  // K维度loop, stride: BK
  for (int k = 0; k < K; k += BK) {
    // 数据搬运, 搬运当前block需要的tile, gmem -> smem
    // 此时的A, B, C已经是当前block需要处理的tile的起始addr
    Asmem[tid_y * BK + tid_x] = A[tid_y * K + tid_x];
    Bsmem[tid_y * BN + tid_x] = B[tid_y * N + tid_x];
    __syncthreads();

    // A_tile x方向loop
    // B_tile y方向loop
    A += BK;
    B += BK * N;

    // tile内求内积
    // A_tile内 y方向并行, x方向loop, stride=1
    // B_tile内 x方向并行, y方向loop, stride=BN
    for (int i = 0; i < BK; i++) {
      tmp += Asmem[tid_y * BK + i] * Bsmem[tid_x + i * BN];
    }
    __syncthreads();
  }
  C[tid_y * N + tid_x] = alpha * tmp + beta * C[tid_y * N + tid_x];
}

#define CEIL_DIV(M, N) (M + N - 1) / N
int main() {
  std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

  // 打开CSV文件
  std::ofstream csv_file("./course5/sgemm_benchmark_v1.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

  for (int N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size);
    float *C_v1 = (float *)malloc(size);

    float *d_A, *d_B, *d_C_v1;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

    bool out_of_memory = false;

    try {
      float alpha = 1.0f;
      float beta = 0.0f;
      std::mt19937 rng(12345);  // 固定 seed，便于复现
      std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
      for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
      }

      checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy A to device failed");
      checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy B to device failed");

      cublasHandle_t handle;
      checkCublasError(cublasCreate(&handle), "cublasCreate failed");

      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

      // warmup
      int warpup_time = 10;  // 热身次数
      for (int i = 0; i < warpup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }

      checkCudaError(cudaEventRecord(stop),
                     "cudaEventRecord(stop cublas) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize cublas failed");

      float cublas_time = 0;
      checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                     "cudaEventElapsedTime cublas failed");

      // 拷贝 cuBLAS 结果
      checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v1
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      dim3 blockDim(1024);
      dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_v1<32>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v1<32>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");

      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v1 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        float ref  = C_cublas[i];
        float got  = C_v1[i];
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
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
      // 写入CSV
      printf("error_count: %d\n", error_count);
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v1);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v1);
    } catch (...) {
      std::cerr << "Out of memory or error during testing size: " << N
                << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory) {
      std::cout << "Finished size: " << N << std::endl;
    } else {
      csv_file << N << ",OOM,OOM,0" << std::endl;
    }
  }

  csv_file.close();

  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
            << std::endl;

  return 0;
}