// 优化思路:
//  一个 warp, 处理 8 个 fragment, 调用多次 wmma
#include <iostream>
#include <cuda_runtime.h>
#include "common/tester.h"
#include "common/common.h"

using namespace nvcuda;

#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel(
  half *A, half *B, half *C, int M, int N, int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WARP_TILE_M * WMMA_TILE_M; // 16 * 4 * 2 = 128
  constexpr int BN = WMMA_N * WARP_TILE_N * WMMA_TILE_N; // 16 * 2 * 4 = 128
  constexpr int BK = WMMA_K;                             // 16
  __shared__ half a_smem[BM][BK], b_smem[BK][BN];        // 128 * 16 * 2 = 4KB

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int warp_m = warp_id / 2; // WMMA_TILE_M: 0,1,2,3
  const int warp_n = warp_id % 2; // WMMA_TILE_N: 0,1

  // a_smem: 128 行, 每行 16 个数据
  //  128 * 16 / 256 = 8, 每个 thread 读取 8 个数据
  int load_a_smem_m = tid / 2;
  int load_a_smem_k = (tid % 2) * 8;

  // b_smem: 16 行, 每行 128 个数据
  //  128 * 16 / 256 = 8, 每个 thread 读取 8 个数据
  int load_b_smem_k = tid / 16;
  int load_b_smem_n = (tid % 16) * 8;

  int load_gmem_m = by * BM + load_a_smem_m;
  int load_gmem_n = bx * BN + load_b_smem_n;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
    C_frag[WARP_TILE_M][WARP_TILE_N];
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
    A_frag[WARP_TILE_M];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
    B_frag[WARP_TILE_N];

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int load_a_gmem_k = k * WMMA_K + load_a_smem_k; // global col of a
        int load_a_gmem_addr = load_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = k * WMMA_K + load_b_smem_k; // global row of b
        int load_b_gmem_addr = load_b_gmem_k * N + load_gmem_n;
        LDST128BITS(b_smem[load_b_smem_k][load_b_smem_n]) =
          (LDST128BITS(B[load_b_gmem_addr]));
        LDST128BITS(a_smem[load_a_smem_m][load_a_smem_k]) =
          (LDST128BITS(A[load_a_gmem_addr]));
        __syncthreads();

        const int warp_a_smem_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &a_smem[warp_a_smem_m][0], BK);
        const int warp_b_smem_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &b_smem[0][warp_b_smem_n], BN);

        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
        by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
        bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j],
                              N,
                              wmma::mem_row_major);
    }
  }
}

void hgemm_wmma_m16n16k16_mma4x2_warp2x4(half *A, half *B, half *C,
                                        int M, int N, int K) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;

  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;

  dim3 block(256);
  dim3 grid(
    div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N),
    div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));

  hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,
    WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char *argv[]) {
  Tester tester(512, 2048, 1024, 1, 10, 100, true);
  tester.evaluate(hgemm_wmma_m16n16k16_mma4x2_warp2x4, "hgemm_wmma_m16n16k16_mma4x2_warp2x4");

  return 0;
}