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

// dim3 blockDim(256);
// dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));
// mysgemm_v2<128, 128, 8, 8, 8>
// BM x BN: 一个block处理的C_tile大小(BM * BN)
// TM x TN: 一个thread处理的C_chunk大小(TM * TN)
// 一个block有128*128/(8*8)=256个thread/chunk -> 确定需要256个thread
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  int tid_num = blockDim.x;
	int tid_x = threadIdx.x;
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  __shared__ float Asmem[BM * BK];
  __shared__ float Bsmem[BK * BN];

  // A([M, K]): 行之间是独立的, A_tile: y方向取tile([BM, BK]), stride: BM * K
  //  loop方向: x轴, loop_stride: BK
  // B([K, N]): 列之间是独立的, B_tile: x方向取tile([BK, BN]), stride: BN * 1
  //  loop方向: y轴, loop_stride: BK * N
	A = &A[bid_y * BM * K];
	B = &B[bid_x * BN];
	C = &C[bid_y * BM * N + bid_x * BN];

	// C: [M, N]
	// C_tile: [BM, BN]   C_chunk: [TM, TN]
  int C_chunks_per_tile_x = BN / TN;

  // 一个block处理一个C_tile(128 * 128), 一个thread处理一个C_chunk(8 * 8)
  // 根据一个block中, 一行有多少个C_chunk, 计算当前待处理chunk的index
	// eg: C_chunks_per_tile_x: 16
	// 	tid_x: 0,  第 0个chunk(0, 0),	 x轴起始index: 0,			 	 y轴起始index: 0
  //  tid_x: 1,  第 1个chunk(0, 1),  x轴起始index: 1  * TN,  y轴起始index: 0
	//  tid_x: 10, 第10个chunk(0, 10), x轴起始index: 10 * TN,  y轴起始index: 0
	//	tid_x: 50, 第50个chunk(3, 2),  x轴起始index: 2  * TN,	 y轴起始index: 3 * TM
  //  tid_x: 255,第255个chunk(15, 15)
  int C_chunk_x_in_block = (tid_x % C_chunks_per_tile_x) * TN;
  int C_chunk_y_in_block = (tid_x / C_chunks_per_tile_x) * TM;

  // 这里是数据搬运, 不是计算粒度, 要将当前block负责的tile, 全部load到smem
	// A_tile: [BM, BK](128, 8)
  // 一共需要load 128*8个元素, 每个thread load一个元素,
  //  优先load一整行, 最多load32行, 所以需要loop
  // A_tile_x性能不好, 一个warp的thread需要访问同一个tile的不同行(不连续)
	int A_tile_x = tid_x % BK; // 0 ~ 7
  int A_tile_y = tid_x / BK; // 0 ~ 31
  int A_tile_loop_stride = tid_num / BK;

	// B_tile: [BK, BN](8, 128)
	int B_tile_x = tid_x % BN; // 0 ~ 127
  int B_tile_y = tid_x / BN; // 0 ~ 1
	int B_tile_loop_stride = tid_num / BN;

	float tmp[TM][TN] = {0.0f};
  #pragma unroll
  for (int k = 0; k < K; k += BK) {
		// 数据流, gmem -> smem(与chunk无关), loop
    #pragma unroll
		for (int i = 0; i < BM; i += A_tile_loop_stride) {
			Asmem[(A_tile_y + i) * BK + A_tile_x] =
				  A[(A_tile_y + i) * K + A_tile_x];
		}
    #pragma unroll
		for (int j = 0; j < BK; j += B_tile_loop_stride) {
			Bsmem[(B_tile_y + j) * BN + B_tile_x] =
				  B[(B_tile_y + j) * N + B_tile_x];
		}
		__syncthreads();

    // 计算流
    // 当前thread, 对A_chunk和B_chunk计算内积
    #pragma unroll
		for (int i = 0; i < BK; ++i) {
      #pragma unroll
			for (int j = 0; j < TM; ++j) {
        #pragma unroll
				for (int l = 0; l < TN; ++l) {
					tmp[j][l] +=
							Asmem[(C_chunk_y_in_block + j) * BK + i] *
              Bsmem[C_chunk_x_in_block + i * BN + l];
				}
			}
		}
		__syncthreads();

		A += BK;
    B += BK * N;
	}

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    #pragma unroll
    for (int j = 0; j < TN; j++)
      C[(C_chunk_y_in_block + i) * N + C_chunk_x_in_block + j] =
          alpha * tmp[i][j] +
          beta * C[(C_chunk_y_in_block + i) * N + C_chunk_x_in_block + j];
  }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
int main() {
  std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

  std::ofstream csv_file("./course5/sgemm_benchmark_v2.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

  for (int N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size);
    float *C_v2 = (float *)malloc(size);

    float *d_A, *d_B, *d_C_v2;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v2, size), "cudaMalloc d_C_v2 failed");

    bool out_of_memory = false;

    try {
      // 初始化矩阵 A 和 B
      float alpha = 1.0f;
      float beta = 0.0f;
      std::mt19937 rng(12345);  // 固定 seed，便于复现
      std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
      for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
      }

      // 拷贝到设备
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
      int warpup_time = 10; // 热身次数
      for (int i = 0; i < warpup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v2, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v2, N),
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
      checkCudaError(cudaMemcpy(C_cublas, d_C_v2, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v2
      checkCudaError(cudaMemset(d_C_v2, 0, size), "cudaMemset d_C_v2 failed");

      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_v2<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v2);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v2, 0, size), "cudaMemset d_C_v2 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v2<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v2);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");
      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v2, d_C_v2, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v2 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        float ref  = C_cublas[i];
        float got  = C_v2[i];
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
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f); // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f); // GFlops
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v2);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v2);

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
