#include <cstdio>
#include <cuda_runtime.h>

template <typename T>
void prefix_sum_CPU(T* in, T* out, int32_t len) {
  for (int i = 0; i < len; ++i) {
    out[i] = in[i];
  }
  for (int i = 1; i < len; ++i) {
    out[i] = out[i - 1] + out[i];
  }
}

// 逐步累加前面距离为stride的元素
template<typename T>
__global__ void koggeStoneScan(T* g_out, T* g_in, int32_t n) {
  extern __shared__ T vec[];

  int tid = threadIdx.x;
  if (tid < n) {
    vec[tid] = g_in[tid];
  } else {
    vec[tid] = 0.0f;
  }
  __syncthreads();

  // vec[i] = sum(vec[0] ~ vec[i-1])
  // stride从1开始, 每次乘2: 1, 2, 4, 8, …
  //  树状扫描
  // tid: 0  1  2  3  4  5  6  7
  // vec: 1  2  3  4  5  6  7  8
  // stride: 1, 每个线程加上前一个元素
  // tid: 0    1    2    3    4    5    6    7
  // vec: 1   2+1  3+2  4+3  5+4  6+5  7+6  8+7  -> tmp=vec[tid]+vec[tid-1]
  // vec: 1    3    5    7    9   11   13    15
  // stride: 2, 每个线程加上前两个元素
  // tid: 0    1    2    3    4    5    6    7
  // vec: 1    3   5+1  7+3 -> tmp=vec[tid]+vec[tid-2]
  // vec: 1    3    6   10   14   18   22   26
  // stride: 4, 每个线程加上前四个元素
  // tid: 0    1    2    3    4    5    6    7
  // vec: 1    3    6   10  14+1 18+3 -> tmp=vec[tid]+vec[tid-4]
  // vec: 1    3    6   10   15   21   28   36
  for (int stride = 1; stride < n; stride *= 2) {
    T tmp = 0;
    if (tid >= stride) {
      tmp = vec[tid] + vec[tid - stride];
    }
    __syncthreads();

    if (tid >= stride) {
      vec[tid] = tmp;
    }
    __syncthreads();
  }

  if (tid < n) {
    g_out[tid] = vec[tid];
  }
}

int main() {
  const int n = 1024;
  const int32_t size = n * sizeof(int);

  int* h_in = reinterpret_cast<int*>(malloc(size));
  int* h_out = reinterpret_cast<int*>(malloc(size));
  int* h_res = reinterpret_cast<int*>(malloc(size));

  for (int i = 0; i < n; ++i) {
    h_in[i] = i;
  }

  prefix_sum_CPU<int>(h_in, h_res, n);

  int* d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  float milliseconds = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  koggeStoneScan<int32_t><<<1, n, size>>>(d_out, d_in, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Kernel execution time: %f ms\n", milliseconds);

  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    if ((h_out[i] != h_res[i]) || (i < 10)) {
      printf("h_out[%d]: %d, h_res[%d]: %d\n", i, h_out[i], i, h_res[i]);
    }
  }

  free(h_in);
  free(h_out);
  free(h_res);

  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}
