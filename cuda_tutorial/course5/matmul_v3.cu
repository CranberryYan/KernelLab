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

// mysgemm_v3<128, 128, 8, 8, 8>
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, float *A, float *B,
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

	// eg: C_chunks_per_tile_x: 16
	// 	tid_x: 0,  第 0个chunk(0, 0),	 x轴起始index: 0,			 	 y轴起始index: 0
  //  tid_x: 1,  第 1个chunk(0, 1),  x轴起始index: 1  * TN,  y轴起始index: 0
	//  tid_x: 10, 第10个chunk(0, 10), x轴起始index: 10 * TN,  y轴起始index: 0
	//	tid_x: 50, 第50个chunk(3, 2),  x轴起始index: 2  * TN,	 y轴起始index: 3 * TM
  //  tid_x: 255,第255个chunk(15, 15)
  int C_chunk_x_in_block = (tid_x % C_chunks_per_tile_x) * TN;
  int C_chunk_y_in_block = (tid_x / C_chunks_per_tile_x) * TM;

  // 这里是数据搬运, 不是计算粒度, 要将当前block负责的tile, 全部load到smem
  // 每个thread一次load 4个元素, BK / 4: 当前这一行, 有多少vec
  //  优先load一整行, load 128行, 无需循环
  //  V2中一个thread load 1个元素, loop 4次, 现在一次load 4个元素, 正好一次全部load
  // A_tile: [BM, BK](128, 8)
  int A_tile_vec4_x = tid_x % (BK / 4); // 0 ~ 1(第几个vec)
  int A_tile_x = A_tile_vec4_x * 4;     // 0, 4
  int A_tile_y = tid_x / (BK / 4);      // 0, 1, ..., 127
  int A_tile_loop_stride = tid_num / (BK / 4); // 128

  // B_tile: [BK, BN](8, 128)
  int B_tile_vec4_x = tid_x % (BN / 4); // 0 ~ 31(第几个vec)
  int B_tile_x = B_tile_vec4_x * 4;     // 0, 4, ..., 124
  int B_tile_y = tid_x / (BN / 4);
  int B_tile_loop_stride = tid_num / (BN / 4); // 8

  float a_frag[TM];
  float b_frag[TN];

	float tmp[TM][TN] = {0.0f};
  #pragma unroll
  for (int k = 0; k < K; k += BK) {
    // 数据流, gmem -> smem(与chunk无关), loop
    #pragma unroll
    for (int i = 0; i < BM; i += A_tile_loop_stride) {
      // 读取时转置, 计算流采用外积(A_chunk的列 * B_chunk的行)
      // Asmem[(A_tile_y + i) * BK + A_tile_x + 0] =
      //     A[(A_tile_y + i) * K + A_tile_x + 0];
      // Asmem[(A_tile_y + i) * BK + A_tile_x + 1] =
      //     A[(A_tile_y + i) * K + A_tile_x + 1];
      // Asmem[(A_tile_y + i) * BK + A_tile_x + 2] =
      //     A[(A_tile_y + i) * K + A_tile_x + 2];
      // Asmem[(A_tile_y + i) * BK + A_tile_x + 3] =
      //     A[(A_tile_y + i) * K + A_tile_x + 3];

      // 频繁访问gmem, 优化到寄存器
      float4 A_vec =
          *reinterpret_cast<float4*>(&A[(A_tile_y + i) * K + A_tile_x]);
      Asmem[(A_tile_x + 0) * BM + A_tile_y + i] = A_vec.x;
      Asmem[(A_tile_x + 1) * BM + A_tile_y + i] = A_vec.y;
      Asmem[(A_tile_x + 2) * BM + A_tile_y + i] = A_vec.z;
      Asmem[(A_tile_x + 3) * BM + A_tile_y + i] = A_vec.w;
    }
    #pragma unroll
    for (int j = 0; j < BK; j += B_tile_loop_stride) {
      float4 B_vec =
          *reinterpret_cast<float4*>(&B[(B_tile_y + j) * N + B_tile_x]);
      *reinterpret_cast<float4*>(&Bsmem[(B_tile_y + j) * BN + B_tile_x]) =
          B_vec;
    }
    __syncthreads();

    // 计算流
    // A_chunk的第i列(此时smem已经是列主序, 所以是连续的)从smem到reg
    #pragma unroll
    for (int i = 0; i < BK; ++i) {
      #pragma unroll
      for (int m = 0; m < TM; m += 4) {
        *reinterpret_cast<float4*>(&a_frag[m]) =
            *reinterpret_cast<float4*>(&Asmem[i * BM + C_chunk_y_in_block + m]);
      }
      #pragma unroll
      for (int n = 0; n < TN; n += 4) {
        *reinterpret_cast<float4*>(&b_frag[n]) =
            *reinterpret_cast<float4*>(&Bsmem[i * BN + C_chunk_x_in_block + n]);
      }

      // 外积
      #pragma unroll
      for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
          tmp[m][n] += a_frag[m] * b_frag[n];
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
    // for (int j = 0; j < TN; j++)
    //   C[(C_chunk_y_in_block + i) * N + C_chunk_x_in_block + j] =
    //       alpha * tmp[i][j] +
    //       beta * C[(C_chunk_y_in_block + i) * N + C_chunk_x_in_block + j];
    for (int j = 0; j < TN; j += 4) {
      float4 ctmp =
          *reinterpret_cast<float4*>(
            &C[(C_chunk_y_in_block + i) * N + C_chunk_x_in_block + j]);
      ctmp.x = alpha * tmp[i][j + 0] + beta * ctmp.x;
      ctmp.y = alpha * tmp[i][j + 1] + beta * ctmp.y;
      ctmp.z = alpha * tmp[i][j + 2] + beta * ctmp.z;
      ctmp.w = alpha * tmp[i][j + 3] + beta * ctmp.w;
      *reinterpret_cast<float4*>(
          &C[(C_chunk_y_in_block + i) * N + C_chunk_x_in_block + j]) = ctmp;
    }
  }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
int main() {
  std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

  std::ofstream csv_file("./course5/sgemm_benchmark_v3.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

  for (int N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size);
    float *C_v3 = (float *)malloc(size);

    float *d_A, *d_B, *d_C_v3;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v3, size), "cudaMalloc d_C_v3 failed");

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
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v3, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v3, N),
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
      checkCudaError(cudaMemcpy(C_cublas, d_C_v3, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v3
      checkCudaError(cudaMemset(d_C_v3, 0, size), "cudaMemset d_C_v3 failed");

      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_v3<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v3);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v3, 0, size), "cudaMemset d_C_v3 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v3<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v3);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");
      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v3, d_C_v3, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v3 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        float ref  = C_cublas[i];
        float got  = C_v3[i];
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

      float ratio = v1_gflops / cublas_gflops;
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << "," << ratio << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v3);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v3);

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
