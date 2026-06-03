#include <cstddef>
#include <cstdio>
#include <random>
#include <sys/types.h>
#include <vector>
#include <ostream>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define TOL 1e-5f
#define BLOCK_SIZE 128
constexpr int WARP_SIZE = 32;

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

// 整个block中的128个thread协作, 把当前tile的
// A_block_tile[BM, BK]和B_block_tile[BK, BN], gmem -> smem
template <const int BM, const int BN, const int BK,
          const int a_row_stride, const int b_row_stride>
__device__ void load_from_gmem(int N, int K,
                               const float* A, const float* B,
                               float* Asmem, float* Bsmem,
                               int a_vec_x, int a_vec_y,
                               int b_vec_x, int b_vec_y) {
  // 为什么不按照V4, 进行一定程度的显示分段pipeline load? gmem -> reg -> smem
  #pragma unroll
  for (uint row = 0; row + a_row_stride <= BM; row += a_row_stride) {
    const float4 tmp = *reinterpret_cast<const float4*>(
        &A[(a_vec_y + row) * K + a_vec_x * 4]);
    Asmem[(a_vec_x * 4 + 0) * BM + a_vec_y + row] = tmp.x;
    Asmem[(a_vec_x * 4 + 1) * BM + a_vec_y + row] = tmp.y;
    Asmem[(a_vec_x * 4 + 2) * BM + a_vec_y + row] = tmp.z;
    Asmem[(a_vec_x * 4 + 3) * BM + a_vec_y + row] = tmp.w;
  }

  #pragma unroll
  for (uint row = 0; row + b_row_stride <= BK; row += b_row_stride) {
    *reinterpret_cast<float4*>(&Bsmem[(b_vec_y + row) * BN + b_vec_x * 4]) =
      *reinterpret_cast<const float4 *>(&B[(b_vec_y + row) * N + b_vec_x * 4]);
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_from_smem(
  float* reg_m, float* reg_n, float* thread_results,
  const float* Asmem, const float* Bsmem, const uint warp_x, const uint warp_y,
  const uint subtile_thread_row, const uint subtile_thread_col) {
  #pragma unroll
  for (uint bk = 0; bk < BK; ++bk) {
    // A_block_tile: [128, 16], A_warp_tile: [64, 16], A_warp_subtile: [64, 16]
    // A_thread_tile: [8, 16], reg_m: [8]
    #pragma unroll
    for (uint w_sub_y_idx = 0; w_sub_y_idx < WMITER; ++w_sub_y_idx) {
      #pragma unroll
      for (uint m = 0; m < TM; ++m) {
        reg_m[w_sub_y_idx * TM + m] =
            Asmem[bk * BM +                       /* block_tile   */
                  warp_y * WM +                   /* warp_tile    */
                  w_sub_y_idx * WSUBM +           /* subwarp_tile */
                  subtile_thread_row * TM +       /* thread_tile  */
                  m];   
      }
    }

    // B_block_tile: [16, 128], B_warp_tile: [16, 64], B_warp_subtile: [16, 16]
    // B_thread_tile: [16, 4], , reg_m: [4 * 4]
    #pragma unroll
    for (uint w_sub_x_idx = 0; w_sub_x_idx < WNITER; ++w_sub_x_idx) {
      #pragma unroll
      for (uint n = 0; n < TN; ++n) {
        reg_n[w_sub_x_idx * TN + n] =
            Bsmem[bk * BN +                       /* block_tile   */
                  warp_x * WN +                   /* warp_tile    */
                  w_sub_x_idx * WSUBN +           /* subwarp_tile */
                  subtile_thread_col * TN +       /* thread_tile  */
                  n];
      }
    }

    // 计算
    #pragma unroll
    for (uint w_sub_y_idx = 0; w_sub_y_idx < WMITER; ++w_sub_y_idx) {
      #pragma unroll
      for (uint w_sub_x_idx = 0; w_sub_x_idx < WNITER; ++w_sub_x_idx) {
        #pragma unroll
        for (uint m = 0; m < TM; ++m) {
          #pragma unroll
          for (uint n = 0; n < TN; ++n) {
            thread_results[(w_sub_y_idx * TM + m) * (WNITER * TN) +
                           (w_sub_x_idx * TN) + n] +=
                reg_m[w_sub_y_idx * TM + m] *
                reg_n[w_sub_x_idx * TN + n];
          }
        }
      }
    }
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) mysgemm_v5(
  int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  // BM, BN = 128, BK = 16
  // 一个block, 负责计算一个C_block_tile[128, 128](先确定C, 后根据C的shape确定A, B)
  // 加载一个A_block_tile[128, 16], B_block_tile[16, 128]
  // K维, 标量loop, 向量加载M, N维, 外积也是用这两个维度
  const uint bid_x = blockIdx.x;
  const uint bid_y = blockIdx.y;
	const uint tid = threadIdx.x;

  // 计算流参数
  // thread_num: 128 -> 4个warp
  // 每个block里的C_block_tile, 分成4部分, warp级并行(C_warp_tile)
  //  A_warp_tile: [64, 16], B_warp_tile: [16, 64], C_warp_tile: [64, 64]
  // warp_x_y: 当前warp负责当前block tile里的哪一个warp tile
  const uint warp_idx = threadIdx.x / WARP_SIZE;
  constexpr uint warp_num_Ctile_x = BN / WN;
  const uint warp_x = warp_idx % warp_num_Ctile_x;
  const uint warp_y = warp_idx / warp_num_Ctile_x;

  // warp_subtile_num:
  //  WM * WN: 一个warp需要处理的元素个数(64 * 64)
  //  WARP_SIZE * TM * TN: 一个warp一次能处理的元素个数(32 * 8 * 4)
  //  所以一共分4段(loop 4次)
  // WMITER: 1, M方向被分成1段
  // WNITER: 4, N方向被分成4段
  // M, N维度共计loop 4次(并行的level是warp_tile, subtile是串行的)
  // A_warp_tile: [64, 16], A_warp_subtile: [64, 16]
  // B_warp_tile: [16, 64], B_warp_subtile: [16, 16]
  // C_warp_tile: [64, 64], C_warp_subtile: [64, 16]
  constexpr uint warp_tile_ele_num = WM * WN;
  constexpr uint warp_subtile_num =
			warp_tile_ele_num / (WARP_SIZE * TM * TN);
  constexpr uint WMITER = warp_subtile_num / WNITER;
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

	constexpr uint thread_tile_m = WMITER * TM;
	constexpr uint thread_tile_n = WNITER * TN;
	constexpr uint thread_tile_ele_num = thread_tile_m * thread_tile_n;

  // subtile_threads_x_num: 在一个C_warp_subtile, x方向需要多少thread
  const uint subtile_threads_x_num = WSUBN / TN;
	const uint lane_id = tid % WARP_SIZE;
	const uint subtile_thread_col =
			lane_id % subtile_threads_x_num; // 当前lane在C_warp_subtile的thread列坐标
	const uint subtile_thread_row =
			lane_id / subtile_threads_x_num; // 当前lane在C_warp_subtile的thread行坐标

	// 综上, 一个C_block_tile: [128, 128], 需要4个C_warp_tile
	//	一个C_warp_tile: [64, 64], 需要C_warp_subtile loop4次
	//	一个C_warp_subtile: [64, 16], 由一个warp负责
	//	一个C_thread_tile: [8, 4], 由一个thread负责
	// 	一个C_warp_subtile: [64, 16]，由32个thread的C_thread_tile拼成
	//  因为一个C_warp_tile需要C_warp_subtile loop4次
	//  thread_tile: 一个thread实际处理的当前C_warp_tile的元素个数

  __shared__ float Asmem[BM * BK];
  __shared__ float Bsmem[BK * BN];

	A = &A[bid_y * BM * K];
  B = &B[bid_x * BN];
  C = &C[bid_y * BM * N + bid_x * BN];

  // 数据流参数
  // 数据流无关warp tile, 只是是把block_tile load到smem
  constexpr uint vec_len = 4;
	constexpr uint a_vec_num_x = BK / vec_len;
	const uint a_vec_x = tid % a_vec_num_x;
	const uint a_vec_y = tid / a_vec_num_x;
	const uint a_row_stride =
			(NUM_THREADS * vec_len) / BK; // 一个block, 一次搬几行

	constexpr uint b_vec_num_x = BN / vec_len;
	const uint b_vec_x = tid % b_vec_num_x;
	const uint b_vec_y = tid / b_vec_num_x;
	const uint b_row_stride = (NUM_THREADS * vec_len) / BN;

	float thread_results[thread_tile_ele_num] = {0.0f};
	float reg_m[thread_tile_m] = {0.f};
	float reg_n[thread_tile_n] = {0.f};

  #pragma unroll
  for (uint bk = 0; bk < K; bk += BK) {
    load_from_gmem<BM, BN, BK, a_row_stride, b_row_stride>(
        N, K, A, B, Asmem, Bsmem, a_vec_x, a_vec_y, b_vec_x, b_vec_y);
    __syncthreads();

    process_from_smem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      reg_m, reg_n, thread_results, Asmem, Bsmem, warp_x, warp_y,
      subtile_thread_row, subtile_thread_col);

    A += BK;
    B += BK * N;
    __syncthreads();
  }

	// 写回
	// C = &C[bid_y * BM * N + bid_x * BN];
	// 此时的C, 对应的是C_block_tile的地址, 以下均需要C_warp_tile的地址
	float* C_warp = C + warp_y * WM * N + warp_x * WN;

	// loop, 一共4次, M方向(y)1次, N方向(x)4次
  #pragma unroll
	for (uint w_sub_y_idx = 0; w_sub_y_idx < WMITER; ++w_sub_y_idx) {
    #pragma unroll
		for (uint w_sub_x_idx = 0; w_sub_x_idx < WNITER; ++w_sub_x_idx) {
			float* C_subwarp =
					C_warp + (w_sub_y_idx * WSUBM) * N + w_sub_x_idx * WSUBN;
      #pragma unroll
			for (uint res_idx_y = 0; res_idx_y < TM; ++res_idx_y) {
        #pragma unroll
				for (uint res_idx_x = 0; res_idx_x < TN; res_idx_x+=4) {
					float4 tmp = *reinterpret_cast<float4*>(
						&C_subwarp[(subtile_thread_row * TM + res_idx_y) * N +
											 subtile_thread_col * TN + res_idx_x]);

					const int i = (w_sub_y_idx * TM + res_idx_y) * TN * WNITER +
                        w_sub_x_idx * TN + res_idx_x;
          tmp.x = alpha * thread_results[i + 0] + beta * tmp.x;
          tmp.y = alpha * thread_results[i + 1] + beta * tmp.y;
          tmp.z = alpha * thread_results[i + 2] + beta * tmp.z;
          tmp.w = alpha * thread_results[i + 3] + beta * tmp.w;
          *reinterpret_cast<float4 *>(
              &C_subwarp[(subtile_thread_row * TM + res_idx_y) * N +
								subtile_thread_col * TN + res_idx_x]) = tmp;
				}
			}
		}
	}
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
int main() {
  std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

  // 打开CSV文件
  std::ofstream csv_file("./course5/sgemm_benchmark_v5.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

  for (int N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size);
    float *C_v5 = (float *)malloc(size);

    float *d_A, *d_B, *d_C_v5;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v5, size), "cudaMalloc d_C_v5 failed");

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
      cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

      // warmup
      int warpup_time = 10; // 热身次数
      for (int i = 0; i < warpup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v5, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 50;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v5, N),
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
      checkCudaError(cudaMemcpy(C_cublas, d_C_v5, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v5
      checkCudaError(cudaMemset(d_C_v5, 0, size), "cudaMemset d_C_v5 failed");

      const uint V5_NUM_THREADS = 128;
      const uint V5_BN = 128;
      const uint V5_BM = 128;
      const uint V5_BK = 16;
      const uint V5_WN = 64;
      const uint V5_WM = 64;
      const uint V5_WNITER = 4;
      const uint V5_TN = 4;
      const uint V5_TM = 8;
      dim3 blockDim(V5_NUM_THREADS);

      constexpr uint NUM_WARPS = V5_NUM_THREADS / 32;

      // warptile in threadblocktile
      static_assert((V5_BN % V5_WN == 0) and (V5_BM % V5_WM == 0));
      static_assert((V5_BN / V5_WN) * (V5_BM / V5_WM) == NUM_WARPS);
      // threads in warpsubtile
      static_assert(
          (V5_WM * V5_WN) % (WARP_SIZE * V5_TM * V5_TN * V5_WNITER) == 0);
      constexpr uint V5_WMITER =
          (V5_WM * V5_WN) / (32 * V5_TM * V5_TN * V5_WNITER);
      // warpsubtile in warptile
      static_assert((V5_WM % V5_WMITER == 0) and (V5_WN % V5_WNITER == 0));

      static_assert(
          (V5_NUM_THREADS * 4) % V5_BK == 0,
          "NUM_THREADS*4 must be multiple of V5_BK to avoid quantization "
          "issues during GMEM->SMEM tiling (loading only parts of the "
          "final row of Bs during each iteraion)");
      static_assert(
          (V5_NUM_THREADS * 4) % V5_BN == 0,
          "NUM_THREADS*4 must be multiple of V5_BN to avoid quantization "
          "issues during GMEM->SMEM tiling (loading only parts of the "
          "final row of As during each iteration)");
      static_assert(
          V5_BN % (16 * V5_TN) == 0,
          "BN must be a multiple of 16*TN to avoid quantization effects");
      static_assert(
          V5_BM % (16 * V5_TM) == 0,
          "BM must be a multiple of 16*TM to avoid quantization effects");
      static_assert((V5_BM * V5_BK) % (4 * V5_NUM_THREADS) == 0,
                    "BM*BK must be a multiple of 4*128 to vectorize loads");
      static_assert((V5_BN * V5_BK) % (4 * V5_NUM_THREADS) == 0,
                    "BN*BK must be a multiple of 4*128 to vectorize loads");

      dim3 gridDim(CEIL_DIV(N, V5_BM), CEIL_DIV(N, V5_BN));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_v5<V5_BM, V5_BN, V5_BK, V5_WM, V5_WN, V5_WNITER,
                   V5_TM, V5_TN, V5_NUM_THREADS>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v5);
      }
      cudaDeviceSynchronize();

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v5) failed");
      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v5<V5_BM, V5_BN, V5_BK, V5_WM, V5_WN, V5_WNITER,
                   V5_TM, V5_TN, V5_NUM_THREADS>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v5);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v5) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v5 failed");
      checkCudaError(cudaGetLastError(), "cuda get last error failed");
      float v5_time = 0;
      checkCudaError(cudaEventElapsedTime(&v5_time, start, stop),
                     "cudaEventElapsedTime v5 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v5, d_C_v5, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v5 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        float ref  = C_cublas[i];
        float got  = C_v5[i];
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
      float v5_gflops =
          repeat_time * 2.0f * N * N * N / (v5_time * 1e6f); // GFlops

      float ratio = v5_gflops / cublas_gflops;
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v5_gflops << ","
               << (error_count == 0 ? "1" : "0") << "," << ratio << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v5);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v5);
      cudaDeviceSynchronize();
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
