
#include <iostream>
#include <cuda_runtime.h>
#include "common/tester.h"
#include "common/common.h"

using namespace nvcuda;

// only 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
template <const int WMMA_M, const int WMMA_N, const int WMMA_K>
__global__ void hgemm_wmma_m16n16k16_naive_kernel(half *A, half *B, half *C,
																									int M, int N, int K) {
	const int NUM_K_TILES = div_ceil(K, WMMA_K);
	const int load_gmem_a_m = blockIdx.y * WMMA_M;
	const int load_gmem_b_n = blockIdx.x * WMMA_N;
	if (load_gmem_a_m >= M && load_gmem_b_n >= N)
		return;

	// wmma::fill_fragment(C_frag, 0.0);
	//	此处定义的 fragment 是该 warp 内共享的(类似与 smem, 是 block 内共享)
	// C = alpha * A x B + beta * C
	// wmma::accumulator: C
  // WMMA_M, WMMA_N, WMMA_K: 每个小块, 要处理的 shape
	// half: 精度(计算时)
	// 当前 frag 中的值
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
	wmma::fill_fragment(C_frag, 0.0);

	// wmma::row_major:
	//	fragemnt 模板的第6个参数 Layout 只在需要描述从/向内存加载(load_matrix_sync)
	// 或存储(store_matrix_sync)时才生效, 而 accumulator 不直接对应某种内存布局,
	// 所以 C_frag 的 Layout 保持默认
	wmma::fragment<
		wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
	wmma::fragment<
		wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

#pragma unroll
	for (int k = 0; k < NUM_K_TILES; ++k) {
		// 数据流
		// ldm:
		//	当前矩阵的 cols_num
		wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
		wmma::load_matrix_sync(B_frag, B + (k * WMMA_K) * N + load_gmem_b_n, N);

		// 计算流
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

		__syncthreads();
	}
	wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N,
													wmma::mem_row_major);
}

void hgemm_wmma_m16n16k16_naive(half *A, half *B, half *C,
																int M, int N, int K) {
	// constexpr int WMMA_M = 32;
	// constexpr int WMMA_N = 8;
	constexpr int WMMA_M = 16;
	constexpr int WMMA_N = 16;
	constexpr int WMMA_K = 16;
	dim3 block(32); // 一个 warp
	dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));
  printf("threadNum: %d, gridDim.x: %d, gridDim.y: %d, loopNum: %d\n",
		(int)32,
		(int)div_ceil(N, WMMA_N),
		(int)div_ceil(M, WMMA_M),
		(int)div_ceil(K, WMMA_K));

	// 切分方式
	//	lhs: 按行切(y), 分 512 / 32 个 block, 每个 block 遍历(x) 1024 / 16 次
	//	rhs: 按列切(x), 分 2048 / 8 个 block, 每个 block 遍历(y) 1024 / 16 次
	hgemm_wmma_m16n16k16_naive_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(
		A, B, C, M, N, K);
}

int main(int argc, char *argv[]) {
	Tester tester(512, 2048, 1024, 1, 10, 100, true);
	tester.evaluate(hgemm_wmma_m16n16k16_naive, "hgemm_wmma_m16n16k16_naive");
	return 0;
}
