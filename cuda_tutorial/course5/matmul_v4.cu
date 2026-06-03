#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>
#include <ostream>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define BLOCK_SIZE 256
#define TOL 1e-3f

void checkCudaError(cudaError_t err, const char* msg) {
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

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 一个block负责[128, 128]，其中一个block有256个thread，一个thread负责[8, 8]
// mysgemm_v4<128, 128, 8, 8, 8>
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(256) mysgemm_v4(int M, int N, int K,
                                                  float alpha,
                                                  float *A, float *B,
                                                  float beta, float *C) {
	constexpr int tid_num = 256;
	int tid_x = threadIdx.x;
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

	// double buffer
  __shared__ float Asmem[2][BM * BK];
  __shared__ float Bsmem[2][BK * BN];

	// A([M, K]): 行之间是独立的, A_tile: y方向取tile([BM, BK]), stride: BM * K
  //  loop方向: x轴, loop_stride: BK
  // B([K, N]): 列之间是独立的, B_tile: x方向取tile([BK, BN]), stride: BN * 1
  //  loop方向: y轴, loop_stride: BK * N
  A = &A[bid_y * BM * K];
  B = &B[bid_x * BN];
  C = &C[bid_y * BM * N + bid_x * BN];

  // C_tile: [BM, BN] C_chunk: [TM, TN]
  const int C_chunks_per_tile_x = BN / TN;

	// eg: C_chunks_per_tile_x: 16
	// 	tid_x: 0,  第 0个chunk(0, 0),	 x轴起始index: 0,			 	 y轴起始index: 0
  //  tid_x: 1,  第 1个chunk(0, 1),  x轴起始index: 1  * TN,  y轴起始index: 0
	//  tid_x: 10, 第10个chunk(0, 10), x轴起始index: 10 * TN,  y轴起始index: 0
	//	tid_x: 50, 第50个chunk(3, 2),  x轴起始index: 2  * TN,	 y轴起始index: 3 * TM
  //  tid_x: 255,第255个chunk(15, 15)
  int C_chunk_x_in_block = (tid_x % C_chunks_per_tile_x) * TN;
  int C_chunk_y_in_block = (tid_x / C_chunks_per_tile_x) * TM;

  constexpr int vec_len = 4;
  constexpr int A_vec_num_per_thread = BK * BM / (tid_num * vec_len);
	int A_tile_vec_num = BK / vec_len; 				  // 实际分配了多少thread在一整行上
	int A_tile_vec4_x = tid_x % A_tile_vec_num; // 0 ~ 1(第几个vec)
  int A_tile_x = A_tile_vec4_x * vec_len;			// 0, 4(thread处理tile的起始index)
	int A_tile_y = tid_x / A_tile_vec_num;			// 0, 1, ..., 127
	int A_tile_loop_stride = tid_num / A_tile_vec_num; // 128

  constexpr int B_vec_num_per_thread = BN * BK / (tid_num * vec_len);
	int B_tile_vec_num = BN / vec_len;				  // 实际分配了多少thread在一整行上
  int B_tile_vec4_x = tid_x % B_tile_vec_num; // 0 ~ 31(第几个vec)
	int B_tile_x = B_tile_vec4_x * vec_len;		  // 0, 4, ..., 124
	int B_tile_y = tid_x / B_tile_vec_num;
	int B_tile_loop_stride = tid_num / B_tile_vec_num; // 8

  float4 A_vec[A_vec_num_per_thread];
  float4 B_vec[B_vec_num_per_thread];

	float a_frag[2][TM];
  float b_frag[2][TN];

	// prologue
	// Ping(smem0)的第一次load tile(gmem -> reg -> smem)
	#pragma unroll
  for (int i = 0; i < BM; i += A_tile_loop_stride) {
    int vec_index = i / A_tile_loop_stride;
		// load时转置(计算流使用外积), 显式使用reg作为中转器
    //  此时没有使用异步copy指令, 所以本质都是gmem -> reg -> smem, 只是显式写了出来
		A_vec[vec_index] =
        *reinterpret_cast<float4*>(&A[(A_tile_y + i) * K + A_tile_x]);
		Asmem[0][OFFSET(A_tile_x + 0, A_tile_y + i, BM)] = A_vec[vec_index].x;
		Asmem[0][OFFSET(A_tile_x + 1, A_tile_y + i, BM)] = A_vec[vec_index].y;
		Asmem[0][OFFSET(A_tile_x + 2, A_tile_y + i, BM)] = A_vec[vec_index].z;
		Asmem[0][OFFSET(A_tile_x + 3, A_tile_y + i, BM)] = A_vec[vec_index].w;
	}

	#pragma unroll
	for (int j = 0; j < BK; j += B_tile_loop_stride) {
    int vec_index = j / B_tile_loop_stride;
    B_vec[vec_index] =
        *reinterpret_cast<float4*>(&B[OFFSET(B_tile_y + j, B_tile_x, N)]);
		*reinterpret_cast<float4*>(&Bsmem[0][OFFSET(B_tile_y + j, B_tile_x, BN)]) =
				B_vec[vec_index];
	}
	__syncthreads();

  // Ping(smem0)的第一次load chunk(smem -> reg[0])
	#pragma unroll
	for (int m = 0; m < TM; m += 4) {
		*reinterpret_cast<float4*>(&a_frag[0][m]) =
			*reinterpret_cast<float4*>(
				&Asmem[0][OFFSET(0, (C_chunk_y_in_block + m), BM)]);
	}

	#pragma unroll
	for (int n = 0; n < TN; n += 4) {
		*reinterpret_cast<float4*>(&b_frag[0][n]) =
			*reinterpret_cast<float4*>(
				&Bsmem[0][OFFSET(0, (C_chunk_x_in_block + n), BN)]);
	}

  // main loop
  // 假设第t个loop
  // smem[load_index]: 当前tile t
  // a_frag[0], b_frag[0]: 当前tile t的bk=0
  // smem[write_index]: 准备给tile t+1使用
  // accum: 已经累加tile 0 ~ tile n-1
  int k = 0;
  int PingPong = 0;
  float tmp[TM][TN] = {0.};

  do {
    k += BK;

    // step1: 预取tile t+1, gmem -> reg
    if (k < K) {
      #pragma unroll
      for (int i = 0; i < BM; i += A_tile_loop_stride) {
        int vec_index = i / A_tile_loop_stride;
        A_vec[vec_index] =
            *reinterpret_cast<float4*>(&A[(A_tile_y + i) * K + k + A_tile_x]);
      }
      #pragma unroll
      for (int j = 0; j < BK; j += B_tile_loop_stride) {
        int vec_index = j / B_tile_loop_stride;
        B_vec[vec_index] =
          *reinterpret_cast<float4*>(&B[OFFSET(k + B_tile_y + j, B_tile_x, N)]);
      }
    }

    // step2: 计算当前tile的bk: 0 ~ BK-2
    #pragma unroll
    for (int bk = 1; bk < BK; bk++) {
      // step2.1: 预取bk=1, 当前frag中已经有bk=0
      #pragma unroll
      for (int m = 0; m < TM; m += 4) {
        *reinterpret_cast<float4*>(&a_frag[bk % 2][m]) =
          *reinterpret_cast<float4*>(
            &Asmem[PingPong][OFFSET(bk, C_chunk_y_in_block + m, BM)]);
      }
      #pragma unroll
      for (int n = 0; n < TN; n += 4) {
        *reinterpret_cast<float4*>(&b_frag[bk % 2][n]) =
          *reinterpret_cast<float4*>(
            &Bsmem[PingPong][OFFSET(bk, C_chunk_x_in_block + n, BN)]);
      }

      // step2.2: 计算, 最后一拍为bk: BK-1 - 1, 差一个bk: BK - 1
      #pragma unroll
      for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
          tmp[m][n] += a_frag[(bk - 1) % 2][m] * b_frag[(bk - 1) % 2][n];
        }
      }
    }

    // step3: 刚才预取的tile t+1, load到smem
    //  所以显式分为两步, step1: gmem -> reg, step2: reg -> smem
    if (k < K) {
      #pragma unroll
      for (int i = 0; i < BM; i += A_tile_loop_stride) {
        int vec_index = i / A_tile_loop_stride;
        // load时转置, 计算流使用外积
        Asmem[!PingPong][OFFSET(A_tile_x + 0, A_tile_y + i, BM)] =
            A_vec[vec_index].x;
        Asmem[!PingPong][OFFSET(A_tile_x + 1, A_tile_y + i, BM)] =
            A_vec[vec_index].y;
        Asmem[!PingPong][OFFSET(A_tile_x + 2, A_tile_y + i, BM)] =
            A_vec[vec_index].z;
        Asmem[!PingPong][OFFSET(A_tile_x + 3, A_tile_y + i, BM)] =
            A_vec[vec_index].w;
      }
      #pragma unroll
      for (int j = 0; j < BK; j += B_tile_loop_stride) {
        int vec_index = j / B_tile_loop_stride;
        *reinterpret_cast<float4*>(
          &Bsmem[!PingPong][OFFSET(B_tile_y + j, B_tile_x, BN)]) =
              B_vec[vec_index];
      }
      __syncthreads();

      // 这是tile t+1, 一定从frag[0]开始
      // 因为BK是8, 最后漏算的是bk: 7, 保存在frag[1]中, 所以不耽误tile t+1的load
      #pragma unroll
      for (int m = 0; m < TM; m += 4) {
          *reinterpret_cast<float4*>(&a_frag[0][m]) =
            *reinterpret_cast<float4*>(
              &Asmem[!PingPong][OFFSET(0, C_chunk_y_in_block + m, BM)]);
      }
      #pragma unroll
      for (int n = 0; n < TN; n += 4) {
          *reinterpret_cast<float4*>(&b_frag[0][n]) =
            *reinterpret_cast<float4*>(
              &Bsmem[!PingPong][OFFSET(0, C_chunk_x_in_block + n, BN)]);
      }

      PingPong = !PingPong;
    }

    // step4: 计算最后一拍bk: BK - 1
    #pragma unroll
    for (int m = 0; m < TM; m++) {
      #pragma unroll
      for (int n = 0; n < TN; n++) {
        tmp[m][n] += a_frag[(BK-1) % 2][m] * b_frag[(BK-1) % 2][n];
      }
    }
  } while (k < K);

  #pragma unroll
  for (int m = 0; m < TM; m++) {
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      float4 ctmp = *reinterpret_cast<float4*>(
        &C[OFFSET(C_chunk_y_in_block + m, C_chunk_x_in_block + n, N)]);
      ctmp.x = alpha * tmp[m][n] + beta * ctmp.x;
      ctmp.y = alpha * tmp[m][n + 1] + beta * ctmp.y;
      ctmp.z = alpha * tmp[m][n + 2] + beta * ctmp.z;
      ctmp.w = alpha * tmp[m][n + 3] + beta * ctmp.w;
      *reinterpret_cast<float4*>(
        &C[OFFSET(C_chunk_y_in_block + m, C_chunk_x_in_block + n, N)]) = ctmp;
    }
  }
}

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// v4: gmem -> reg -> smem
// v4_1: 异步搬运
//  问题: B是可以的, 因为外积, B矩阵拿一行, 连续;但是A矩阵, 需要拿一列, 不连续

// 一个block负责[128, 128]，其中一个block有256个thread，一个thread负责[8, 8]
// mysgemm_v4_1<128, 128, 8, 8, 8>
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(256) mysgemm_v4_1(int M, int N, int K,
                                                    float alpha,
                                                    const float *__restrict__ A,
                                                    const float *__restrict__ B,
                                                    float beta,
                                                    float *__restrict__ C) {
	constexpr int tid_num = 256;
	int tid_x = threadIdx.x;
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

	// double buffer
  __shared__ float Asmem[2][BM * BK];
  __shared__ float Bsmem[2][BK * BN];

	// A([M, K]): 行之间是独立的, A_tile: y方向取tile([BM, BK]), stride: BM * K
  //  loop方向: x轴, loop_stride: BK
  // B([K, N]): 列之间是独立的, B_tile: x方向取tile([BK, BN]), stride: BN * 1
  //  loop方向: y轴, loop_stride: BK * N
  A = &A[bid_y * BM * K];
  B = &B[bid_x * BN];
  C = &C[bid_y * BM * N + bid_x * BN];

  // C_tile: [BM, BN] C_chunk: [TM, TN]
  const int C_chunks_per_tile_x = BN / TN;

	// eg: C_chunks_per_tile_x: 16
	// 	tid_x: 0,  第 0个chunk(0, 0),	 x轴起始index: 0,			 	 y轴起始index: 0
  //  tid_x: 1,  第 1个chunk(0, 1),  x轴起始index: 1  * TN,  y轴起始index: 0
	//  tid_x: 10, 第10个chunk(0, 10), x轴起始index: 10 * TN,  y轴起始index: 0
	//	tid_x: 50, 第50个chunk(3, 2),  x轴起始index: 2  * TN,	 y轴起始index: 3 * TM
  //  tid_x: 255,第255个chunk(15, 15)
  int C_chunk_x_in_block = (tid_x % C_chunks_per_tile_x) * TN;
  int C_chunk_y_in_block = (tid_x / C_chunks_per_tile_x) * TM;

  constexpr int vec_len = 4;
	int A_tile_vec_num = BK / vec_len; 				  // 实际分配了多少thread在一整行上
	int A_tile_vec4_x = tid_x % A_tile_vec_num; // 0 ~ 1(第几个vec)
  int A_tile_x = A_tile_vec4_x * vec_len;			// 0, 4(thread处理tile的起始index)
	int A_tile_y = tid_x / A_tile_vec_num;			// 0, 1, ..., 127
	int A_tile_loop_stride = tid_num / A_tile_vec_num; // 128

	int B_tile_vec_num = BN / vec_len;				  // 实际分配了多少thread在一整行上
  int B_tile_vec4_x = tid_x % B_tile_vec_num; // 0 ~ 31(第几个vec)
	int B_tile_x = B_tile_vec4_x * vec_len;		  // 0, 4, ..., 124
	int B_tile_y = tid_x / B_tile_vec_num;
	int B_tile_loop_stride = tid_num / B_tile_vec_num; // 8

	float a_frag[2][TM];
  float b_frag[2][TN];

	// prologue
	// Ping(smem0)的第一次load tile(gmem -> reg -> smem)
	#pragma unroll
  for (int i = 0; i < BM; i += A_tile_loop_stride) {
    const float* gmem_a = &A[(A_tile_y + i) * K + A_tile_x];

    __pipeline_memcpy_async(
      &Asmem[0][OFFSET(A_tile_x + 0, A_tile_y + i, BM)],
      gmem_a + 0,
      4);
    __pipeline_memcpy_async(
      &Asmem[0][OFFSET(A_tile_x + 1, A_tile_y + i, BM)],
      gmem_a + 1,
      4);
    __pipeline_memcpy_async(
      &Asmem[0][OFFSET(A_tile_x + 2, A_tile_y + i, BM)],
      gmem_a + 2,
      4);
    __pipeline_memcpy_async(
      &Asmem[0][OFFSET(A_tile_x + 3, A_tile_y + i, BM)],
      gmem_a + 3,
      4);
	}

	#pragma unroll
	for (int j = 0; j < BK; j += B_tile_loop_stride) {
    __pipeline_memcpy_async(
      &Bsmem[0][OFFSET(B_tile_y + j, B_tile_x, BN)],
      &B[OFFSET(B_tile_y + j, B_tile_x, N)],
      16
    );
	}

  __pipeline_commit();
  __pipeline_wait_prior(0);
	__syncthreads();

  // Ping(smem0)的第一次load chunk(smem -> reg[0])
	#pragma unroll
	for (int m = 0; m < TM; m += 4) {
		*reinterpret_cast<float4*>(&a_frag[0][m]) =
			*reinterpret_cast<float4*>(
				&Asmem[0][OFFSET(0, (C_chunk_y_in_block + m), BM)]);
	}

	#pragma unroll
	for (int n = 0; n < TN; n += 4) {
		*reinterpret_cast<float4*>(&b_frag[0][n]) =
			*reinterpret_cast<float4*>(
				&Bsmem[0][OFFSET(0, (C_chunk_x_in_block + n), BN)]);
	}

  // main loop
  // 假设第t个loop
  // smem[load_index]: 当前tile t
  // a_frag[0], b_frag[0]: 当前tile t的bk=0
  // smem[write_index]: 准备给tile t+1使用
  // accum: 已经累加tile 0 ~ tile n-1
  int k = 0;
  int PingPong = 0;
  float tmp[TM][TN] = {0.};

  do {
    k += BK;

    // step1: 预取tile t+1, gmem -> reg
    if (k < K) {
      #pragma unroll
      for (int i = 0; i < BM; i += A_tile_loop_stride) {
        const float* gmem_a =
          &A[(A_tile_y + i) * K + k + A_tile_x];

        __pipeline_memcpy_async(
          &Asmem[!PingPong][OFFSET(A_tile_x + 0, A_tile_y + i, BM)],
          gmem_a + 0,
          4);
        __pipeline_memcpy_async(
          &Asmem[!PingPong][OFFSET(A_tile_x + 1, A_tile_y + i, BM)],
          gmem_a + 1,
          4);
        __pipeline_memcpy_async(
          &Asmem[!PingPong][OFFSET(A_tile_x + 2, A_tile_y + i, BM)],
          gmem_a + 2,
          4);
        __pipeline_memcpy_async(
          &Asmem[!PingPong][OFFSET(A_tile_x + 3, A_tile_y + i, BM)],
          gmem_a + 3,
          4);
      }

      #pragma unroll
      for (int j = 0; j < BK; j += B_tile_loop_stride) {
        __pipeline_memcpy_async(
            &Bsmem[!PingPong][OFFSET(B_tile_y + j, B_tile_x, BN)],
            &B[OFFSET(k + B_tile_y + j, B_tile_x, N)],
            16
        );
      }
      __pipeline_commit();
    }

    // step2: 计算当前tile的bk: 0 ~ BK-2
    #pragma unroll
    for (int bk = 1; bk < BK; bk++) {
      int frag_load = bk & 1;
      int frag_compute = (bk - 1) & 1;
      // step2.1: 预取bk=1, 当前frag中已经有bk=0
      #pragma unroll
      for (int m = 0; m < TM; m += 4) {
        *reinterpret_cast<float4*>(&a_frag[frag_load][m]) =
          *reinterpret_cast<float4*>(
            &Asmem[PingPong][OFFSET(bk, C_chunk_y_in_block + m, BM)]);
      }
      #pragma unroll
      for (int n = 0; n < TN; n += 4) {
        *reinterpret_cast<float4*>(&b_frag[frag_load][n]) =
          *reinterpret_cast<float4*>(
            &Bsmem[PingPong][OFFSET(bk, C_chunk_x_in_block + n, BN)]);
      }

      // step2.2: 计算, 最后一拍为bk: BK-1 - 1, 差一个bk: BK - 1
      #pragma unroll
      for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
          tmp[m][n] =
              fmaf(a_frag[frag_compute][m], b_frag[frag_compute][n], tmp[m][n]);
        }
      }
    }

    // step3: 刚才预取的tile t+1, load到smem
    //  所以显式分为两步, step1: gmem -> reg, step2: reg -> smem
    if (k < K) {
      __pipeline_wait_prior(0);
      __syncthreads();

      // 这是tile t+1, 一定从frag[0]开始
      // 因为BK是8, 最后漏算的是bk: 7, 保存在frag[1]中, 所以不耽误tile t+1的load
      #pragma unroll
      for (int m = 0; m < TM; m += 4) {
          *reinterpret_cast<float4*>(&a_frag[0][m]) =
            *reinterpret_cast<float4*>(
              &Asmem[!PingPong][OFFSET(0, C_chunk_y_in_block + m, BM)]);
      }
      #pragma unroll
      for (int n = 0; n < TN; n += 4) {
          *reinterpret_cast<float4*>(&b_frag[0][n]) =
            *reinterpret_cast<float4*>(
              &Bsmem[!PingPong][OFFSET(0, C_chunk_x_in_block + n, BN)]);
      }

      PingPong = !PingPong;
    }

    // step4: 计算最后一拍bk: BK - 1
    #pragma unroll
    for (int m = 0; m < TM; m++) {
      #pragma unroll
      for (int n = 0; n < TN; n++) {
        tmp[m][n] += a_frag[(BK-1) % 2][m] * b_frag[(BK-1) % 2][n];
      }
    }
  } while (k < K);

  #pragma unroll
  for (int m = 0; m < TM; m++) {
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      float4 ctmp = *reinterpret_cast<float4*>(
        &C[OFFSET(C_chunk_y_in_block + m, C_chunk_x_in_block + n, N)]);
      ctmp.x = alpha * tmp[m][n] + beta * ctmp.x;
      ctmp.y = alpha * tmp[m][n + 1] + beta * ctmp.y;
      ctmp.z = alpha * tmp[m][n + 2] + beta * ctmp.z;
      ctmp.w = alpha * tmp[m][n + 3] + beta * ctmp.w;
      *reinterpret_cast<float4*>(
        &C[OFFSET(C_chunk_y_in_block + m, C_chunk_x_in_block + n, N)]) = ctmp;
    }
  }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

int main() {
  std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

  std::ofstream csv_file("./course5/sgemm_benchmark_v4.csv");
  csv_file <<
      "Size,CUBLAS_GFLOPS,MySGEMM_v4_FLOPS,MySGEMM_v4_1_FLOPS,\
       Matched,Ratio4,Ratio4_1" << std::endl;

  for (auto N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float* A = reinterpret_cast<float*>(malloc(size));
    float* B = reinterpret_cast<float*>(malloc(size));
    float* C_cublas = reinterpret_cast<float*>(malloc(size));
    float* C_v4 = reinterpret_cast<float*>(malloc(size));
    float* C_v4_1 = reinterpret_cast<float*>(malloc(size));

    float *d_A, *d_B, *d_C_v4;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v4, size), "cudaMalloc d_C_v4 failed");

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
      int warm_time = 10;
      for (int i = 0; i < warm_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v4, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 100;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v4, N),
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
      checkCudaError(cudaMemcpy(C_cublas, d_C_v4, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v1
      checkCudaError(cudaMemset(d_C_v4, 0, size), "cudaMemset d_C_v1 failed");

      // 一个block负责[128, 128]，其中一个block有256个thread，一个thread负责[8, 8]
      dim3 blockDim(BLOCK_SIZE);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

			for (int i = 0; i < warm_time; ++i) {
        mysgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(N, N, N, alpha,
																														 d_A, d_B, beta,
																														 d_C_v4);
			}
      cudaDeviceSynchronize();
      checkCudaError(cudaGetLastError(), "mysgemm_v4 warmup failed");

      checkCudaError(cudaEventRecord(start), "cudaEventRecord(start v4) failed");
			for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(N, N, N, alpha,
																														 d_A, d_B, beta,
																														 d_C_v4);
			}
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v4) failed");
      checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize v4 failed");
      checkCudaError(cudaGetLastError(), "mysgemm_v4 launch failed");
      
      float v4_time = 0;
      checkCudaError(cudaEventElapsedTime(&v4_time, start, stop),
                     "cudaEventElapsedTime v4 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v4, d_C_v4, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v4 failed");

			for (int i = 0; i < warm_time; ++i) {
        mysgemm_v4_1<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(N, N, N, alpha,
                                                               d_A, d_B, beta,
                                                               d_C_v4);
			}
      cudaDeviceSynchronize();
      checkCudaError(cudaGetLastError(), "mysgemm_v4_1 warmup failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v4_1) failed");
			for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v4_1<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(N, N, N, alpha,
																														   d_A, d_B, beta,
																														   d_C_v4);
			}
      checkCudaError(cudaEventRecord(stop),
                     "cudaEventRecord(stop v4_1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v4_1 failed");
      checkCudaError(cudaGetLastError(), "mysgemm_v4_1 launch failed");

      float v4_1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v4_1_time, start, stop),
                     "cudaEventElapsedTime 4_1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v4_1, d_C_v4, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v4_1 failed");

      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        float ref  = C_cublas[i];
        float got  = C_v4_1[i];
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
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);

      float v4_gflops =
          repeat_time * 2.0f * N * N * N / (v4_time * 1e6f);

      float v4_1_gflops =
          repeat_time * 2.0f * N * N * N / (v4_1_time * 1e6f);

      float ratio_v4 = v4_gflops / cublas_gflops;
      float ratio_v4_1 = v4_1_gflops / cublas_gflops;

      // 写入CSV
      csv_file << N << ","
               << cublas_gflops << ","
               << v4_gflops << ","
               << v4_1_gflops << ","
               << (error_count == 0 ? "1" : "0") << ","
               << ratio_v4 << ","
               << ratio_v4_1 << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v4);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v4);
      free(C_v4_1);
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
