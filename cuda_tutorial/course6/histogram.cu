#include <random>
#include <iostream>
#include <cuda_runtime.h>

__global__ void histogram(int* input, int* hist, int n, int low, int high) {
  int thread_num = gridDim.x * blockDim.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < n; i += thread_num) {
    int val = input[i];
    if (val >= low && val < high) {
      val -= low;
      atomicAdd(&hist[val], 1);
    }
  }
}

int main() {
  int M = 128;
  int N = 4096;
  int low = -1000;
  int high = 128;
  int size = M * N;
  int length = high - low;
  int* input = reinterpret_cast<int*>(malloc(size * sizeof(int)));

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(low-1, high);
  for (int i = 0; i < size; ++i) {
    int input_tmp = static_cast<int>(dist_float(mt));
    input[i] = input_tmp;
  }

  int h_hist[length] = {};
  for (int i = 0; i < size; ++i) {
    if (input[i] >= low && input[i] < high) {
      h_hist[input[i] - low] += 1;
    }
  }

  int grid_size = 512;
  int block_size = 512;

  int *d_input;
  int *d_hist;
  cudaMalloc(&d_input, size * sizeof(int));
  cudaMalloc(&d_hist, length * sizeof(int));
  cudaMemset(d_hist, 0, length * sizeof(int));
  cudaMemcpy(d_input, input, sizeof(int) * size, cudaMemcpyHostToDevice);
  histogram<<<grid_size, block_size>>>(d_input, d_hist, size, low, high);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cuda error:%d\n", err);
  }

  int res[length];
  cudaMemcpy(res, d_hist, length * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < length; ++i) {
    if (i < 10 || (res[i] != h_hist[i])) {
      printf("GPU: %d : %d\n", i, res[i]);
      printf("CPU: %d : %d\n", i, h_hist[i]);
    }
  }

  free(input);
  cudaFree(d_input);

  return 0;
}
