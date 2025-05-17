#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void ncu_add_scalar(float* x, float* y, float* out) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // 访存量:
  //  读x, y 写out
  //  load: 1024 * 8192 * 2 * bpe = 67108864Bytes
  //  store: 1024 * 8192 * 1 * bpe = 33554432Bytes
  //  1024 * 8192 * 3 * bpe = 100663296Bytes
  // 计算量:
  //  1024 * 8192 = 8388608
  // GPU峰值算力(fp32): 29.8TFLOPS
  // GPU峰值带宽: 760GB/s
  // bound:
  //  计算量 / 峰值算力 = 8388608 / 29.8TFLOPS = 281496 / 10^12 = 2.814 * 10^(-7)s
  //  访存量 / 峰值带宽 = 100663296Bytes / 760GB/s = 132451 / 10^9 = 1.324 * 10(-4)s
  // mem bound
  // 指令数:
  //  load: 8192 * 1024 / 32 * 2 = 524888Inst
  //  store: 8192 * 1024 / 32 * 1 = 262144Inst
  // sector:
  //  load: 67108864Bytes / 32Byes = 2097152Sectors
  //        2097152Sectors / 524888Inst = 4Rectors/Req
  //  store: 33554432Bytes / 32Bytes = 1048576Sectors
  //         1048576Sectors / 262144Inst = 4Rectors/Req
  out[gid] = x[gid] + y[gid];
}

__global__ void ncu_add_vec(float* x, float* y, float* out) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // 访存量:
  //  读x, y 写out
  //  load: 1024 * 8192 * 2 * bpe = 67108864Bytes
  //  store: 1024 * 8192 * 1 * bpe = 33554432Bytes
  //  1024 * 8192 * 3 * bpe = 100663296Bytes
  // 计算量:
  //  1024 * 8192 = 8388608
  // 依然是mem bound, 因为vec读取不会影响相关计算
  // 指令数:
  //  load: 8192 * 1024 / 32 / 4 * 2 = 131072Inst
  //  strore: 8192 * 1024 / 32 / 4 = 65536Inst
  // sector:
  //  load: 67108864Bytes / 32Byes = 2097152Sectors
  //        2097152Sectors / 131072Inst = 16Rectors/Req
  //  store: 33554432Bytes / 32Bytes = 1048576Sectors
  //         1048576Sectors / 65536Inst = 16Rectors/Req
  float4* x_vec = reinterpret_cast<float4*>(x);
  float4* y_vec = reinterpret_cast<float4*>(y);
  float4* out_vec = reinterpret_cast<float4*>(out);

  out_vec[gid].x = x_vec[gid].x + y_vec[gid].x;
  out_vec[gid].y = x_vec[gid].y + y_vec[gid].y;
  out_vec[gid].z = x_vec[gid].z + y_vec[gid].z;
  out_vec[gid].w = x_vec[gid].w + y_vec[gid].w;
}

int main() {
  int bpe = sizeof(float);
  int shape = 1024 * 8192;

  float* x = reinterpret_cast<float*>(malloc(bpe * shape));
  float* y = reinterpret_cast<float*>(malloc(bpe * shape));
  float* out = reinterpret_cast<float*>(malloc(bpe * shape));

  float* x_d = nullptr;
  float* y_d = nullptr;
  float* out_d = nullptr;
  cudaMalloc(&x_d, bpe * shape);
  cudaMalloc(&y_d, bpe * shape);
  cudaMalloc(&out_d, bpe * shape);

  for (int i = 0; i < shape; ++i) {
    x[i] = i % 128;
    y[i] = i % 256;
  }

  cudaMemcpy(x_d, x, bpe * shape, cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, bpe * shape, cudaMemcpyHostToDevice);

  // scalar
  int blockNum_scalar = 8192;
  int threadNum_scalar = 1024;
  dim3 grid(blockNum_scalar);
  dim3 block(threadNum_scalar);

  ncu_add_scalar<<<grid, block>>>(x_d, y_d, out_d);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr
      << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaDeviceSynchronize();

  cudaMemcpy(out, out_d, bpe * shape, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; ++i) {
      cout << out[i] << " ";
  }
  cout << endl;

  // vec
  int blockNum_vec = 8192;
  int threadNum_vec  = 1024 / 4;
  dim3 grid_v(blockNum_vec);
  dim3 block_v(threadNum_vec);
  ncu_add_vec<<<grid_v, block_v>>>(x_d, y_d, out_d);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr
      << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaDeviceSynchronize();

  cudaMemcpy(out, out_d, bpe * shape, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; ++i) {
      cout << out[i] << " ";
  }
  cout << endl;

  free(x);
  free(y);
  free(out);
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(out_d);

  return 0;
}
