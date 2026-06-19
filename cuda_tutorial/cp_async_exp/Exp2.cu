// Exp2: 故意破坏wait / sync, 观察cp.async的completion与visibility

#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(call)                                                \
	do {                                                                  \
		cudaError_t err = (call);                                         	\
		if (err != cudaSuccess) {                                          	\
			fprintf(stderr, "CUDA error at %s:%d: %s\n",                  		\
							__FILE__, __LINE__, cudaGetErrorString(err));          		\
			std::exit(EXIT_FAILURE);                                       		\
		}                                                                 	\
	} while (0)

__device__ __forceinline__ void delay_loop(int iters) {
	int x = 0;
	#pragma unroll 1
	for (int i = 0; i < iters; ++i) {
			x += i;
	}
	asm volatile("" :: "r"(x));
}

/*
Variant 0:
	正确版：self read + wait + __syncthreads
	每个线程读自己搬到 smem[tid] 的数据

Variant 1:
	错误版：self read + no wait + no sync
	cp.async 发起后立刻读 smem[tid], 可能读到旧值

Variant 2:
	通常正确：self read + wait + no sync
	每个线程只消费自己发起的 copy, 不跨线程读
	这个实验用于说明：如果没有跨线程消费, __syncthreads 不一定是必要条件

Variant 3:
	错误版：cross-warp read + wait + no sync
	每个线程读 smem[tid ^ 32], 也就是读另一个 warp 搬的数据
	wait 只等自己线程的 copy, 不等别的线程的 copy
	为了让错误更稳定, 奇数 warp 在发起 cp.async 前故意 delay

Variant 4:
	正确版：cross-warp read + wait + __syncthreads
	和 Variant 3 一样跨 warp 读, 但加了 __syncthreads
*/

template<int BLOCK_SIZE, int VARIANT>
__global__ void cp_async_exp2_kernel(const float* __restrict__ in,
																		 float* __restrict__ out,
																		 int N) {
  int tid = threadIdx.x;
	int warp_id = tid >> 5;
  int bid = blockIdx.x;
  int gid = bid * BLOCK_SIZE + tid;

  __shared__ float smem[BLOCK_SIZE];

	// 先把 smem 初始化成明显的非法值
	// 如果 cp.async 没完成就读, 很容易读到这些 sentinel
	smem[tid] = -100000.0f - static_cast<float>(tid);

	__syncthreads();

	// Variant 3/4 用来测试跨 warp 消费
	// 故意让奇数 warp 晚一点发起 copy
	// 这样如果没有 __syncthreads, 偶数 warp 很可能读到奇数 warp 的旧 smem
	if constexpr (VARIANT == 3 || VARIANT == 4) {
		if (warp_id & 1) {
			delay_loop(20000);
		}
	}

	if (gid < N) {
		__pipeline_memcpy_async(&smem[tid], &in[gid], sizeof(float));
	}
	__pipeline_commit();

	if constexpr (VARIANT == 0 || VARIANT == 2 || VARIANT == 3 || VARIANT == 4) {
		__pipeline_wait_prior(0);
	}

	if constexpr (VARIANT == 0 || VARIANT == 4) {
		__syncthreads();
	}

	int read_tid;
	if constexpr (VARIANT == 3 || VARIANT == 4) {
		// 跨 warp 读
		// 0 <-> 1 warp, 2 <-> 3 warp, ...
		read_tid = tid ^ 32;
	} else {
		// 自己读自己
		read_tid = tid;
	}

	if (gid < N) {
		out[gid] = smem[read_tid];
	}
}

template<int BLOCK_SIZE, int VARIANT>
void launch_kernel(const float* d_in, float* d_out, int N) {
	int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	cp_async_exp2_kernel<BLOCK_SIZE, VARIANT><<<grid, BLOCK_SIZE>>>(d_in, d_out, N);
}

struct CheckResult {
	int mismatches;
	int first_idx[8];
	float first_expected[8];
	float first_got[8];
};

CheckResult check_self_read(const std::vector<float>& in,
													  const std::vector<float>& out,
													  int N) {
	CheckResult r{};
	r.mismatches = 0;

	for (int i = 0; i < 8; ++i) {
		r.first_idx[i] = -1;
		r.first_expected[i] = 0.0f;
		r.first_got[i] = 0.0f;
	}

	for (int i = 0; i < N; ++i) {
		float expected = in[i];
		float got = out[i];

		if (std::fabs(expected - got) > 1e-6f) {
			if (r.mismatches < 8) {
				int k = r.mismatches;
				r.first_idx[k] = i;
				r.first_expected[k] = expected;
				r.first_got[k] = got;
			}
			r.mismatches++;
		}
	}

	return r;
}

template<int BLOCK_SIZE>
CheckResult check_cross_warp_read(const std::vector<float>& in,
																  const std::vector<float>& out,
																  int N) {
	CheckResult r{};
	r.mismatches = 0;

	for (int i = 0; i < 8; ++i) {
		r.first_idx[i] = -1;
		r.first_expected[i] = 0.0f;
		r.first_got[i] = 0.0f;
	}

	for (int gid = 0; gid < N; ++gid) {
		int block_base = (gid / BLOCK_SIZE) * BLOCK_SIZE;
		int tid = gid % BLOCK_SIZE;
		int read_tid = tid ^ 32;
		int expected_gid = block_base + read_tid;

		float expected = in[expected_gid];
		float got = out[gid];

		if (std::fabs(expected - got) > 1e-6f) {
			if (r.mismatches < 8) {
				int k = r.mismatches;
				r.first_idx[k] = gid;
				r.first_expected[k] = expected;
				r.first_got[k] = got;
			}
			r.mismatches++;
		}
	}

	return r;
}

void print_result(const char* name, const CheckResult& r) {
	printf("%-48s : %s, mismatches = %d\n",
					name,
					r.mismatches == 0 ? "PASS" : "FAIL",
					r.mismatches);

	if (r.mismatches > 0) {
		int n = std::min(r.mismatches, 8);
		for (int i = 0; i < n; ++i) {
			printf("    first mismatch %d: idx = %d, expected = %.6f, got = %.6f\n",
							i,
							r.first_idx[i],
							r.first_expected[i],
							r.first_got[i]);
		}
	}
}

int main() {
	constexpr int BLOCK_SIZE = 256;

	// 为了 cross-warp check 简单, N 取 BLOCK_SIZE 的整数倍
	int N = 8 * 1024 * 1024;
	size_t bytes = static_cast<size_t>(N) * sizeof(float);

	int device = 0;
	CHECK_CUDA(cudaGetDevice(&device));

	cudaDeviceProp prop{};
	CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

	printf("GPU                  = %s\n", prop.name);
	printf("Compute capability   = %d.%d\n", prop.major, prop.minor);
	printf("N                    = %d\n", N);
	printf("BLOCK_SIZE           = %d\n", BLOCK_SIZE);
	printf("Input bytes          = %.2f MB\n\n", bytes / 1024.0 / 1024.0);

	std::vector<float> h_in(N);
	std::vector<float> h_out(N);

	for (int i = 0; i < N; ++i) {
		h_in[i] = static_cast<float>((i % 10007) + 1);
	}

	float* d_in = nullptr;
	float* d_out = nullptr;

	CHECK_CUDA(cudaMalloc(&d_in, bytes));
	CHECK_CUDA(cudaMalloc(&d_out, bytes));

	CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

	// Variant 0
	CHECK_CUDA(cudaMemset(d_out, 0, bytes));
	launch_kernel<BLOCK_SIZE, 0>(d_in, d_out, N);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
	auto r0 = check_self_read(h_in, h_out, N);

	// Variant 1
	CHECK_CUDA(cudaMemset(d_out, 0, bytes));
	launch_kernel<BLOCK_SIZE, 1>(d_in, d_out, N);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
	auto r1 = check_self_read(h_in, h_out, N);

	// Variant 2
	CHECK_CUDA(cudaMemset(d_out, 0, bytes));
	launch_kernel<BLOCK_SIZE, 2>(d_in, d_out, N);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
	auto r2 = check_self_read(h_in, h_out, N);

	// Variant 3
	CHECK_CUDA(cudaMemset(d_out, 0, bytes));
	launch_kernel<BLOCK_SIZE, 3>(d_in, d_out, N);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
	auto r3 = check_cross_warp_read<BLOCK_SIZE>(h_in, h_out, N);

	// Variant 4
	CHECK_CUDA(cudaMemset(d_out, 0, bytes));
	launch_kernel<BLOCK_SIZE, 4>(d_in, d_out, N);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
	auto r4 = check_cross_warp_read<BLOCK_SIZE>(h_in, h_out, N);

	printf("==== Exp2: cp.async wait / sync correctness ====\n");
	print_result("V0 CORRECT: self read, wait + sync", r0);
	print_result("V1 BUG:     self read, no wait, no sync", r1);
	print_result("V2 OK:      self read, wait, no sync", r2);
	print_result("V3 BUG:     cross-warp read, wait, no sync", r3);
	print_result("V4 CORRECT: cross-warp read, wait + sync", r4);

	printf("\nExpected interpretation:\n");
	printf("V0 should PASS.\n");
	printf("V1 should usually FAIL because cp.async is not waited before smem read.\n");
	printf("V2 should PASS because each thread reads only its own copied data.\n");
	printf("V3 should usually FAIL because each thread reads data produced by another warp, but there is no CTA sync.\n");
	printf("V4 should PASS because wait + __syncthreads protects cross-thread shared-memory consumption.\n");

	printf("\nNote:\n");
	printf("V1 and V3 are intentionally invalid programs. If they occasionally PASS, that does not make them semantically correct.\n");
	printf("V3 includes an artificial odd-warp delay to make the missing __syncthreads bug easier to observe.\n");

	CHECK_CUDA(cudaFree(d_in));
	CHECK_CUDA(cudaFree(d_out));

	return 0;
}