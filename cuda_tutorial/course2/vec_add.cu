#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK_SIZE 1024 // 一个block中包含1024个thread

__global__ void vec_add(int *A, int *B, int *C, int N) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  if (gid > N) {
    return;
  }

  for (int i = gid; i < N; i += gridDim.x) {
    C[i] = A[i] + B[i];
  }
}


int main() {
  int N = 16384;
  size_t size = N * sizeof(int);

  int* A = reinterpret_cast<int*>(malloc(size));
  int* B = reinterpret_cast<int*>(malloc(size));
  int* C = reinterpret_cast<int*>(malloc(size));
  int* res = reinterpret_cast<int*>(malloc(size));

  for (int i = 0; i < N; ++i) {
    A[i] = i;
    B[i] = i * 2;
    C[i] = -1;
    res[i] = A[i] + B[i];
  }

  int* d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

  int numBlocks = std::max(((N + BLOCK_SIZE - 1) / BLOCK_SIZE), 1024);
  printf("array size:%d\n", N);
  printf("blocks:%d\n", numBlocks);
  printf("thread num per block:%d\n", BLOCK_SIZE);

  vec_add<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    if (C[i] != res[i]) {
      printf("wrong!!!\n");
      printf("C[%d]: %d, res[%d]: %d\n", i, C[i], i, res[i]);
    } else if (i < 10) {
      printf("C[%d]: %d, res[%d]: %d\n", i, C[i], i, res[i]);
    }
  }

  free(A);
  free(B);
  free(C);
  free(res);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);


  return 0;
}