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
// 先归约(up-sweep), 后分发(down-sweep)
template<typename T>
__global__ void workEfficientScan(T* g_out, T* g_in, int32_t n) {
  extern __shared__ T vec[];

  int32_t tid = threadIdx.x;
  int BLOCK_SIZE = blockDim.x;
  int ele_per_block = BLOCK_SIZE;

  if (tid < n) {
    vec[tid] = g_in[tid];
  } else {
    vec[tid] = 0;
  }
  __syncthreads();

  // Up-sweep: 归约树向上构建
  int stride = 1;
  for (; stride < ele_per_block; stride *=2) {
    // stride: 1
    // index: 1, 3, 5, 7, 9, ...
    // vec[1] += vec[0]
    // vec[3] += vec[2]
    // vec[5] += vec[4]
    // vec[7] += vec[8]
    //  步长为1的相邻两个元素归约, index: 1, 3, 5, 7进行更新
    //  index: 1, 已求得前缀和
    // stride: 2
    // index: 3, 7, 11, 15, ...
    // vec[3] += vec[1]
    // vec[7] += vec[5]
    //  步长为2的相邻两个元素归约, index: 3, 7
    //  index: 3, 已求得前缀和
    // stride: 4
    // index: 7
    // vec[7] += vec[3]
    //  步长为4的相邻两个元素归约, index: 7
    // 最右边的元素是所有元素的和, 循环终止

    // [1, 2, 3, 4, 5, 6, 7, 8]
    // s1: [1, 3, 3, 7, 5, 11, 7, 15]
    // s3: [1, 3, 3, 10, 5, 11, 7, 26]
    // s4: [1, 3, 3, 10, 5, 11, 7, 36]

    //                                vec[7]
    //                vec[3](左半边的所有和)           vec[7](右半边的所有和)
    //        vec[1]          vec[3]          vec[5]          vec[7](所以下扫要从倒数第二次开始, 且只处理右边)
    //    vec[0]  vec[1]  vec[2]  vec[3]  vec[4]  vec[5]  vec[6]  vec[7]
    // 左边的节点, 全部处理好了
    // stride: 每个半区的长度(vec[1] += vec[0], 实际是处理了区间长度为2的元素)
    // * 2: 整个区间
    // - 1: 将大区间末尾位置转换为0-based下标, 因此index是该大区间的右端点
    int index = tid * stride * 2 - 1;
    if (index < ele_per_block && (index - stride) >= 0) {
      vec[index] += vec[index - stride];
    }
    __syncthreads();
  }

  // Down-sweep: 把部分和传播下去
  // / 4: 倒数第二轮
  stride = ele_per_block / 4;
  for(; stride > 0; stride /= 2) {
    // 把左半部分的累积和, 加到右半部分对应的节点上
    // [1, 2, 3, 4, 5, 6, 7, 8]
    // up-sweep:
    // s1: [1, 3, 3, 7, 5, 11, 7, 15]
    // s3: [1, 3, 3, 10, 5, 11, 7, 26]
    // s4: [1, 3, 3, 10, 5, 11, 7, 36]
    // down-sweep:
    // [1, 3, 3, 10, 5, 11, 7, 36]
    // s2: index: 3   stride: 2 -> vec[5] += vec[3]
    //     index: 7   stride: 2
    // [1, 3, 3, 10, 5, 21, 7, 36]
    // s1: index: 1   stride: 1 -> vec[2] += vec[1]
    //     index: 3   stride: 1 -> vec[4] += vec[3]
    //     index: 5   stride: 1 -> vec[6] += vec[5]
    int index = tid * stride * 2 - 1;
    if (index + stride < ele_per_block && index > 0) {
      // 左边已经全处理完了, 需要处理右边
      vec[index + stride] += vec[index];
    }
    __syncthreads();
  }

  if (tid < n) g_out[tid] = vec[tid];
}

template <typename T>
__global__ void scan_and_collect_sums(T* g_out, T* g_in, T* g_sums, int n) {
  extern __shared__ T vec[];
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int BLOCK_SIZE = blockDim.x;
  int global_idx_base = bid * BLOCK_SIZE;

  if (global_idx_base + tid < n) {
    vec[tid] = g_in[global_idx_base + tid];
  } else {
    vec[tid] = 0;
  }
  __syncthreads();

  // up-sweep
  int stride = 1;
  for (; stride < BLOCK_SIZE; stride *= 2) {
    int index = tid * stride * 2 - 1;
    if (index < BLOCK_SIZE && (index - stride) >= 0) {
      vec[index] += vec[index - stride];
    }
    __syncthreads();
  }

  // down-sweep
  stride /= 4;
  for (; stride > 0; stride /= 2) {
    int index = tid * stride * 2 - 1;
    if (index + stride < BLOCK_SIZE && index > 0) {
      vec[index + stride] += vec[index];
    }
    __syncthreads();
  }

  if (global_idx_base + tid < n) {
    g_out[global_idx_base + tid] = vec[tid];
  }
  if (tid == BLOCK_SIZE - 1) {
    g_sums[bid] = vec[BLOCK_SIZE - 1];
  }
}

template <typename T>
__global__ void add_scanned_sums(T* g_out, T* g_sum, int n) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  if (bid == 0) {
    // 前BLOCK_SIZE个元素, 已经是前缀和
    return;
  }

  int BLOCK_SIZE = blockDim.x;
  int global_idx_base = bid * BLOCK_SIZE;
  T add_val = g_sum[bid - 1];

  if (global_idx_base + tid < n) {
    g_out[global_idx_base + tid] += add_val;
  }
}

int main() {
  const int n = 1021 * 63;
  const int thread_per_block = 1024;
  const int numBlocks = (n + thread_per_block - 1) / thread_per_block;
  const int32_t size = n * sizeof(int);
  const int32_t smem_size = thread_per_block * sizeof(int);

  int* h_in = reinterpret_cast<int*>(malloc(size));
  int* h_out = reinterpret_cast<int*>(malloc(size));
  int* h_res = reinterpret_cast<int*>(malloc(size));

  for (int i = 0; i < n; ++i) {
    h_in[i] = i;
  }

  prefix_sum_CPU<int>(h_in, h_res, n);

  int* d_in, *d_out, *d_sums;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);
  cudaMalloc(&d_sums, numBlocks * sizeof(int));

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  float milliseconds = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  scan_and_collect_sums<int32_t><<<numBlocks, thread_per_block, smem_size>>>(
    d_out, d_in, d_sums, n);
  workEfficientScan<int32_t><<<1, thread_per_block, smem_size>>>(
    d_sums, d_sums, numBlocks);
  add_scanned_sums<int32_t><<<numBlocks, thread_per_block>>>(d_out, d_sums, n);
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
  cudaFree(d_sums);

  return 0;
}
