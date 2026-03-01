#include <chrono>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

// shuffle
// __shfl_sync
//  广播, 将某个lane中的数据快速分发给warp中的其他thread
__global__ void test_shfl_bradocast(int* out, int* in, const int src_lane) {
  int val = in[blockIdx.x * blockDim.x + threadIdx.x];

  // 2, 34, 66, ..., (n * 32 + 2)
  val = __shfl_sync(0xFFFFFFFF, val, src_lane);
  out[blockIdx.x * blockDim.x + threadIdx.x] = val;
}

int main() {
  const int block_num = 2;
  const int block_size = 64;
  const int src_lane = 2;

  int in_h[block_num * block_size];
  int out_h[block_num * block_size];

  for (int i = 0; i < block_num; ++i) {
    for (int j = 0; j < block_size; ++j) {
      in_h[i * block_size + j] = j;
    }
  }

  int* in_d, *out_d;
  cudaMalloc((void**)&in_d, block_num * block_size * sizeof(int));
  cudaMalloc((void**)&out_d, block_num * block_size * sizeof(int));

  cudaMemcpy(in_d, in_h,
             block_num * block_size * sizeof(int), cudaMemcpyHostToDevice);
  test_shfl_bradocast<<<block_num, block_size>>>(out_d, in_d, src_lane);
  cudaMemcpy(out_h, out_d,
             block_num * block_size * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < block_num * block_size; ++i) {
    printf("%d: %d\n", i, out_h[i]);
  }


  return 0;
}