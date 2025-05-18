// 优化思路:
//  问题: naive 中的切块太小 -> 数据搬运次数过多,
// 但是 tensor core 限制了每次计算的 frag 的大小
//  解决措施: 借助 smem, 先把数据预存到 smem, 然后计算的时候
// 一个 block 中分不同的 warp 还是按照 tensor core 要求的大小来计算
// M: 512, N: 2048, K: 1024
// block_size: 256(8 个 warp) 8 * 16 * 16 -> 4 * 2 * (16 * 16)
// tile_size: M: [64(512 / 8(gridDim.y)), 16(1024 / 64(遍历 64 次))]
//            N: [16(1024 / 64(遍历 64 次)), 32(2048 / 64(gridDim.x))]
// grid_size: [2048 / 32, 512 / 64, 1] -> [64, 8, 1]
// smem: lhs, 搬运 64 * 16 个元素
//       rhs, 搬运 16 * 32 个元素
#include <iostream>
#include <cuda_runtime.h>
#include "common/tester.h"
#include "common/common.h"

using namespace nvcuda;

#define WARP_SIZE 32
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])

// m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(half *A, half *B, half *C,
                                                   int M, int N, int K) {
  const int NUM_K_TILES = div_ceil(K, WMMA_K); // loop_nums
  constexpr int BM = WMMA_M * WMMA_TILE_M; // 64
  constexpr int BN = WMMA_N * WMMA_TILE_N; // 32
  constexpr int BK = WMMA_K;

  __shared__ half a_smem[BM][BK];
  __shared__ half b_smem[BK][BN];

  // 保证相同的 warp 下 thread 执行相同的指令
  const int bx = blockIdx.x; // rhs_tile_id
  const int by = blockIdx.y; // lhs_tile_id
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int warp_m = warp_id / 2; // 0, 1, 2, 3
  const int warp_n = warp_id % 2; // 0, 1

  // 数据流
  // 256 个 thread 分别 load a_smem: 64 * 16, b_smem: 16 * 32
  // a_smem: 64 * 16 / 256 -> 4(一次 load 4 个元素)
  //  一行需要 4 个 thread, 一共 64 行
  // b_smem: 16 * 32 / 256 -> 2(一次 load 2 个元素)
  //  一行需要 16 个 thread, 一共 16 行
  const int load_a_smem_m = tid / 4; // 0 - 63
  const int load_a_smem_k = (tid % 4) * 4; // 0, 4, 8, 12
  const int load_b_smem_k = tid / 16; // 0 - 15
  const int load_b_smem_n = (tid % 16) * 2; // 0, 2, 4, ..., 32
  const int load_a_gmem_m = by * BM + load_a_smem_m;
  const int load_b_gmem_n = bx * BN + load_b_smem_n;

  wmma::fragment<
    wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
  wmma::fill_fragment(C_frag, 0.0);

  wmma::fragment<
    wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
  wmma::fragment<
    wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

  if (load_a_gmem_m >= M && load_b_gmem_n >= N)
    return;

#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    int load_gmem_a_k = k * WMMA_K + load_a_smem_k;
    int load_gmem_a_addr = load_a_gmem_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_b_smem_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_b_gmem_n;

    LDST64BITS(a_smem[load_a_smem_m][load_a_smem_k]) =
      (LDST64BITS(A[load_gmem_a_addr]));
    LDST32BITS(b_smem[load_b_smem_k][load_b_smem_n]) =
      (LDST32BITS(B[load_gmem_b_addr]));
    __syncthreads();

    wmma::load_matrix_sync(A_frag, &a_smem[warp_m * WMMA_M][0], BK);
    wmma::load_matrix_sync(B_frag, &b_smem[0][warp_n * WMMA_N], BN);

    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

    __syncthreads();
  }

  const int store_c_gmem_m = by * BM + warp_m * WMMA_M;
  const int store_c_gmem_n = bx * BN + warp_n * WMMA_N;
  wmma::store_matrix_sync(C + store_c_gmem_m * N + store_c_gmem_n,
                          C_frag,
                          N,
                          wmma::mem_row_major);
}

void hgemm_wmma_m16n16k16_mma4x2(half* A, half* B, half* C,
                                int M, int N, int K) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;

  dim3 block(256);
  dim3 grid(
    div_ceil(N, WMMA_N * WMMA_TILE_N),
    div_ceil(M, WMMA_M * WMMA_TILE_M));

  hgemm_wmma_m16n16k16_mma4x2_kernel<
    WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N><<<grid, block>>>(
      A, B, C, M, N, K);
}

int main() {
  Tester tester(512, 2048, 1024, 1, 10, 100, true);
  tester.evaluate(hgemm_wmma_m16n16k16_mma4x2, "hgemm_wmma_m16n16k16_mma4x2");

  return 0;
}