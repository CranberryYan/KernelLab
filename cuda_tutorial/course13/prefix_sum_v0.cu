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
  T* src = vec;
  T* dst = vec + n;

  int32_t tid = threadIdx.x;

  if (tid < n) {
    src[tid] = g_in[tid];
  } else {
    src[tid] = 0;
  }
  __syncthreads();

  // V0
  // for (int stride = 1; stride < n; stride *= 2) {
  //   T tmp = 0;
  //   if (tid >= stride) {
  //     tmp = vec[tid] + vec[tid - stride];
  //   }
  //   __syncthreads();

  //   if (tid >= stride) {
  //     vec[tid] = tmp;
  //   }
  //   __syncthreads();
  // }

  for (int stride = 1; stride < n; stride *= 2) {
    if (tid >= stride) {
      dst[tid] = src[tid] + src[tid - stride];
    } else {
      dst[tid] = src[tid];
    }
    __syncthreads();

    // 更新src
    T* temp = src;
    src = dst;
    dst = temp;
  }

  if (tid < n) {
    g_out[tid] = src[tid];
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
  koggeStoneScan<int32_t><<<1, n, size*2>>>(d_out, d_in, n);
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
