#include <chrono>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

// shuffle
// __shfl_up_sync
// 数据向上平移
// 在一个warp中, lane_id为t的thread将从lane_id为t - delta的thread中读取val的值,
//  若t - delta < 0, 则保留自身的val
// 0 1 2 3 ... 31
// A B C D ... Z
// A B A B ... X(delta为2, AB保留(t - delta < 0))
__global__ void test_shfl_up(int* out, int* in, const int delta) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  int val = in[gid];
  val = __shfl_up_sync(0xFFFFFFFF, val, delta);
  out[gid] = val;
}

int main() {
  const int block_num = 2;
  const int block_size = 64;
  const int delta = 2;

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
  test_shfl_up<<<block_num, block_size>>>(out_d, in_d, delta);
  cudaMemcpy(out_h, out_d,
             block_num * block_size * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < block_num * block_size; ++i) {
    printf("%d: %d\n", i, out_h[i]);
  }


  return 0;
}