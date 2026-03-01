#include <chrono>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) do {                                      \
  cudaError_t _e = (call);                                         \
  if (_e != cudaSuccess) {                                         \
    fprintf(stderr, "[CUDA] %s:%d: %s (%d)\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(_e), (int)_e);  \
    std::exit(1);                                                  \
  }                                                                \
} while (0)

bool compare_results(const float* cpu, const float* gpu, int N, int C,
                     float epsilon = 1e-5f) {
  for (int i = 0; i < N * C; ++i) {
    if (i < 10) {
      std::cout << "index " << i << ": CPU=" << cpu[i]
                << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                << std::endl;
    }
    if (fabs(cpu[i] - gpu[i]) > epsilon) {
      std::cout << "Difference at index " << i << ": CPU=" << cpu[i]
                << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

// 计算公式: Oi = e^(Zi-Z) / sum(eZj-Z)
// input: N x C
// output: N x C
// 操作维度: 1, 对这一行的每一个元素(列), 进行softmax, 找出每一行的最大值
void safe_softmax_forward_CPU(float* out, const float* in, int N, int C) {
  for (int i = 0; i < N; ++i) {
    // 每一行的起始地址
    const float* in_row = in + i * C;
    float* out_row = out + i * C;

    // 查找最大值
    float maxval = -INFINITY;
    for (int j = 0; j < C; ++j) {
      if (in_row[j] > maxval) {
        maxval = in_row[j];
      }
    }

    // 求解分子, 分母
    float sum = 0.0f;
    for (int j = 0; j < C; ++j) {
      out_row[j] = expf(in_row[j]-maxval);
      sum += out_row[j];
    }

    // 计算输出
    float norm = 1.0f / sum;
    for (int j = 0; j < C; ++j) {
      out_row[j] *= norm;
    }
  }
}

// 操作维度: -1(压缩原始输入维度为[N, C], 固定操作-1维度)
// 配置N个block, 为每一行分配一个block,
//  该block内的thread(固定为1)共同完成改行的规约求解max
__global__ void safe_softmax_forward_GPU_v0(float* out, const float* in,
                                            int N, int C) {
  // 与CPU对照, i表示行号, j表示列号
  int i = blockIdx.x;

  if (i < N) {
    const float* in_row = in + i * C;
    float* out_row = out + i * C;

    // 查找最大值(优化点, 可以使用smem, 由一个thread进行查找, 其余共享)
    float maxval = -INFINITY;
    for (int j = 0; j < C; ++j) {
      if (in_row[j] > maxval) {
        maxval = in_row[j];
      }
    }

    // 计算分子, 分母(优化点, sum和out使用迭代的方式进行求解 -> FLA)
    float sum = 0.0f;
    for (int j = 0; j < C; ++j) {
      out_row[j] = expf(in_row[j] - maxval);
      sum += out_row[j];
    }

    // 计算输出
    float norm = 1.0f / sum;
    for (int j = 0; j < C; ++j) {
      out_row[j] *= norm;
    }
  }
}

// 优化: 使用smem
__global__ void safe_softmax_forward_GPU_v1(float* out, const float* in,
                                            int N, int C) {
  // __shared__ float smem[256]; // 静态, 在编译器确定
  extern __shared__ float smem[]; // 动态分配, 在kernel调用时指定
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;

	if (bid < N) {
		const float* in_row = in + bid * C;
		float* out_row = out + bid * C;

		// 查找最大值
		// eg: C: 1024, block_size: 128
		// 每个thread会不重叠的在8(1024/128)个数据中找出相对的最大值
		// smem[0]: 是C[0]~C[7]的最大值, smem[1]: 是C[8]~C[15]的最大值
		// 规约, 找出这128(block_size)中的最大值
		// 0~63与64~127比较 -> 0~31与32~63比较 ...
		// 这个过程中, 也不是所有thread都工作, 在0~31与32~63比较时, 64~127闲置
		// 最后的maxval会产生在0~1的比较中
		float maxval_tmp = -INFINITY;
		for (int i = tid; i < C; i += block_size) {
			maxval_tmp = fmaxf(maxval_tmp, in_row[i]);
		}
		smem[tid] = maxval_tmp;
		__syncthreads();

		for (int stride = block_size / 2; stride >= 1; stride /= 2) {
			if (tid < stride) {
				smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
			}
			__syncthreads();
		}

		float maxval = smem[0];

		// 计算分子, 分母
		for (int i = tid; i < C; i += block_size) {
			out_row[i] = expf(in_row[i] - maxval);
		}
		__syncthreads();

		// sum与max求解过程相同
		float sum_tmp = 0.0f;
		for (int i = tid; i < C; i += block_size) {
			sum_tmp += out_row[i];
		}

		smem[tid] = sum_tmp; // 不会再使用max, 所以可以复用
		__syncthreads();

		for (int stride = block_size / 2; stride >= 1; stride /= 2) {
			if (tid < stride) {
				smem[tid] += smem[tid + stride];
			}
			__syncthreads();
		}

		float sum = smem[0];
		float norm = 1.0f / sum;
		for (int i = tid; i < C; i += block_size) {
			out_row[i] *= norm;
		}
	}
}

__device__ float warpReduceMax(float val) {
  //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  // delta: 16
  // 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  // 输出: lane0 ~ 15保存着当前warp中较大的16个元素

  // delta: 8
  //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  //  8  9 10 11 12 13 14 15 16  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  // 输出: lane0 ~ 7保存着当前waro中较大的8个元素

  // delta: 4
  //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  //  4  5  6  7  8  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  // 输出: lane0 ~ 3保存着当前waro中较大的4个元素

  // delta: 2
  //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  //  2  3  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  // 输出: lane0 ~ 1保存着当前waro中较大的2个元素

  // delta: 1
  //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  //  1  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  // 输出: lane0保存着当前waro中最大的元素

  // lane0: 整个warp的max
  int lane = threadIdx.x & 31;
  for (int stride = 16; stride >= 1; stride >>= 1) {
    unsigned mask = __activemask();
    float other = __shfl_down_sync(mask, val, stride);
    if (lane < stride) val = fmaxf(val, other);
  }
  return val; // lane0 是整个 warp max
}

__device__ float warpReduceSum(float val) {
  int lane = threadIdx.x & 31;
  for (int stride = 16; stride >= 1; stride >>= 1) {
    unsigned mask = __activemask();
    float other = __shfl_down_sync(mask, val, stride);
    if (lane < stride) val += other;
  }
  return val; // lane0 是整个 warp max
}

// blockDim.x: 固定为32
// 优化点: blockDim.x固定为32
__global__ void safe_softmax_forward_GPU_v2(float* out, const float* in,
                                            int N, int C) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid < N) {
    const float* in_row = in + bid * C;
		float* out_row = out + bid * C;

		// 查找最大值
    float maxval = -INFINITY;
    float max_tmp = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
      // 让这32个thread, 找到32个max_tmp
      max_tmp = fmaxf(max_tmp, in_row[i]);
    }

    // blockDim.x固定为32, 因此找到这个warp中的max -> 该row的max
    maxval = warpReduceMax(max_tmp);

    // 1. tid0存入smem
    // if (tid == 0) {
    //   smem[0] = maxval;
    // }
    // __syncthreads();

    // 2. 继续使用shfl进行广播
    maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);

		// 计算分子, 分母
    float sum = 0.0f;
    float sum_tmp = 0.0f;
		for (int i = tid; i < C; i += blockDim.x) {
			out_row[i] = expf(in_row[i] - maxval);
      sum_tmp += out_row[i];
		}
    sum = warpReduceSum(sum_tmp);
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);

		float norm = 1.0f / sum;
		for (int i = tid; i < C; i += blockDim.x) {
			out_row[i] *= norm;
		}
  }
}

// blockDim.x: 32 * n
// smem进行不同warp之间的数据交互
__global__ void safe_softmax_forward_GPU_v3(float* out, const float* in,
                                            int N, int C) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x % 32;
  int warp_per_block = (blockDim.x + 31) >> 5;

  // smem划分:
  //  每个warp使用shfl求出该warp中的max和sum
  //  smem中保存warp_per_block个maxval和sumval
  //  smem: 0 ~ warp_per_block-1: maxval
  //        warp_per_block ~ warp_per_block + warp_per_block - 1: sumval
  extern __shared__ float smem[];
  float* smem_max = smem;
  float* smem_sum = smem + warp_per_block;

  for (int row = bid; row < N; row += gridDim.x) {
    const float* in_row = in + row * C;
		float* out_row = out + row * C;

		// 查找最大值
    float maxval = -INFINITY;
    float max_tmp = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
      // 每个thread找到n(C / blockDim.x)个值中的最大值
      max_tmp = fmaxf(max_tmp, in_row[i]);
    }

    // 找到每个warp中的max
    max_tmp = warpReduceMax(max_tmp);

    if (lane_id == 0) {
      smem_max[warp_id] = max_tmp;
    }
    __syncthreads();

    // tid0求解warp_per_block个warp中的最大值
    #if 0
    // 1. tid0 loop求解
    if (tid == 0) {
      for (int i = 0; i < warp_per_block; ++i) {
        maxval = fmaxf(maxval, smem_max[i]);
      }

      smem_max[0] = maxval;
    }
    __syncthreads();
    #else
    // 2. 继续规约, 每个block最多32个thread, 一定能在两次规约内求解block中最大值
    if (warp_id == 0) {
      max_tmp = (lane_id < warp_per_block) ? smem_max[lane_id] : -INFINITY;
      max_tmp = warpReduceMax(max_tmp);
      if (lane_id == 0) {
        smem_max[0] = max_tmp;
      }
    }
    __syncthreads();
    #endif
    maxval = smem_max[0];

    // 计算分子, 分母
    float sum = 0.0f;
    float sum_tmp = 0.0f;

    for (int i = tid; i < C; i += blockDim.x) {
      // 每个thread求解n(C / blockDim.x)个值的和
			out_row[i] = __expf(in_row[i] - maxval);
      sum_tmp += out_row[i];
    }

    sum_tmp = warpReduceSum(sum_tmp);
    if (lane_id == 0) {
      smem_sum[warp_id] = sum_tmp;
    }
    __syncthreads();

    // tid0求解warp_per_block个值的和
    #if 0
    // 1. tid0 loop求解
    if (tid == 0) {
      for (int i = 0; i < warp_per_block; ++i) {
        sum += smem_sum[i];
      }
    }
    #else
    // 2. 继续规约
    if (warp_id == 0) {
      sum_tmp = (lane_id < warp_per_block) ? smem_sum[lane_id] : 0;
      sum_tmp = warpReduceSum(sum_tmp);
      if (lane_id == 0) {
        smem_sum[0] = sum_tmp;
      }
    }
    __syncthreads();
    #endif
    sum = smem_sum[0];

    float norm = __fdividef(1.0f, sum);

    for (int i = tid; i < C; i += blockDim.x) {
      out_row[i] *= norm;
    }
  }
}

__device__ __forceinline__ void from_HBM_to_smem(float* smem,
                                                 const int tid,
                                                 const float* row_tile,
                                                 int tile_cols,
                                                 bool aligned16) {
  int col = tid;
  int stride = blockDim.x;
  if (aligned16) {
    float4* smem4 = reinterpret_cast<float4*>(smem);
    const float4* row4 = reinterpret_cast<const float4*>(row_tile);

    // 数据搬运: 显存 -> smem
    // 这里最多分配32KB, 8KB个FP32
    int C4 = tile_cols >> 2;

    // C: 8192, C4: 2048, blockDim.x: 1024 -> 循环2次 -> 手动2次循环展开
    for (; col + stride < C4; col += 2 * stride) {
      smem4[col] = row4[col];
      smem4[col + stride] = row4[col + stride];
    }

    // 处理剩余 0/1 次
    for (; col < C4; col += stride) {
      smem4[col] = row4[col];
    }

    for (int col = (C4 << 2) + tid; col < tile_cols; col += blockDim.x) {
      smem[col] = row_tile[col];
    }
  } else {
    for (; col + stride < tile_cols; col += 2 * stride) {
      smem[col] = row_tile[col];
      smem[col + stride] = row_tile[col + stride];
    }

    // 处理剩余 0/1 次
    for (; col < tile_cols; col += stride) {
      smem[col] = row_tile[col];
    }
  }
}

template<bool is_tile>
__global__ void safe_softmax_forward_GPU_v4(float* __restrict__ out,
                                            const float* __restrict__ in,
                                            int N, int C,
                                            int smem_size) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_per_block = (blockDim.x + 31) >> 5;
  if (warp_per_block > 32) return;

  // smem划分:
  // 动态: 保存每一列, 防止去显存进行访存
  // 静态: warp_size个sum和max
  extern __shared__ __align__(16) unsigned char smem_raw[];
  float* smem = reinterpret_cast<float*>(smem_raw);
  float4* smem4 = reinterpret_cast<float4*>(smem);
  __shared__ float smem_max[32];
  __shared__ float smem_sum[32];
  __shared__ float smem_tile_max_scalar;
  __shared__ float smem_tile_sum_scalar;

  for (int row = bid; row < N; row += gridDim.x) {
    float maxval = -INFINITY;
    float sum = 0.0f;
    if (tid == 0) {
      smem_tile_max_scalar = -INFINITY; smem_tile_sum_scalar = 0.0f;
    }
    __syncthreads();

    const float* in_row = in + row * C;
    float* out_row = out + row * C;
    bool aligned16_in = (((uintptr_t)in_row) & 0x000F) == 0;
    bool aligned16_out = (((uintptr_t)out_row) & 0x000F) == 0;

    const int tile_size = smem_size;
    const int tile_elems = tile_size / sizeof(float);
    int tile_num = (C + tile_elems - 1) / tile_elems;
    if constexpr (!is_tile) {
      if (tile_num > 1) return;
    }

    // 数据搬运&查找最大值
    #if 1
    for (int t = 0; t < tile_num; ++t) {
      int col = tid;
      int stride = blockDim.x;
      float tile_max_tmp = -INFINITY;
      const float* in_row_tile = in_row + t * tile_elems;
      int tile_cols = min(tile_elems, C - t*tile_elems);

      from_HBM_to_smem(smem, tid, in_row_tile, tile_cols, aligned16_in);
      __syncthreads();

      int C4 = tile_cols >> 2;
      for (; col + stride < C4; col += 2 * stride) {
        float4 v0 = smem4[col];
        float4 v1 = smem4[col + stride];
        float m0 = fmaxf(fmaxf(v0.x, v0.y), fmaxf(v0.z, v0.w));
        float m1 = fmaxf(fmaxf(v1.x, v1.y), fmaxf(v1.z, v1.w));
        tile_max_tmp = fmaxf(tile_max_tmp, fmaxf(m0, m1));
      }
      for (; col < C4; col += stride) {
        float4 v = smem4[col];
        float m = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        tile_max_tmp = fmaxf(tile_max_tmp, m);
      }
      for (int col = (C4 << 2) + tid; col < tile_cols; col += blockDim.x) {
        tile_max_tmp = fmaxf(tile_max_tmp, smem[col]);
      }

      // 找到每个warp中的max
      tile_max_tmp = warpReduceMax(tile_max_tmp);
      if (lane_id == 0) {
        smem_max[warp_id] = tile_max_tmp;
      }
      __syncthreads();

      // tid0求解当前block中的最大值
      // 此时求出的最大值是当前tile的最大值
      if (warp_id == 0) {
        tile_max_tmp = (lane_id < warp_per_block) ? smem_max[lane_id] : -INFINITY;
        tile_max_tmp = warpReduceMax(tile_max_tmp);
        if (lane_id == 0) {
          smem_tile_max_scalar = fmaxf(smem_tile_max_scalar, tile_max_tmp);
        }
      }
      __syncthreads();
    }
    maxval = smem_tile_max_scalar;
    #endif

    // 计算分子, 分母
    #if 1
    for (int t = 0; t < tile_num; ++t) {
      int col = tid;
      int stride = blockDim.x;
      float tile_sum_tmp = 0.0f;
      const float* in_row_tile = in_row + t * tile_elems;
      float* out_row_tile = out_row + t * tile_elems;
      int tile_cols = min(tile_elems, C - t*tile_elems);
      if (aligned16_in && aligned16_out) {
        const float4* in_row4 = reinterpret_cast<const float4*>(in_row_tile);
        float4* out_row4 = reinterpret_cast<float4*>(out_row_tile);
        int C4 = tile_cols >> 2;

        if constexpr (is_tile) {
          for (; col + stride < C4; col += 2 * stride) {
            float4 b0 = in_row4[col];
            float4 b1 = in_row4[col + stride];
            float4 v0 = make_float4(__expf(b0.x - maxval),
                                    __expf(b0.y - maxval),
                                    __expf(b0.z - maxval),
                                    __expf(b0.w - maxval));
            float4 v1 = make_float4(__expf(b1.x - maxval),
                                    __expf(b1.y - maxval),
                                    __expf(b1.z - maxval),
                                    __expf(b1.w - maxval));
            tile_sum_tmp += (v0.x + v0.y) + (v0.z + v0.w);
            tile_sum_tmp += (v1.x + v1.y) + (v1.z + v1.w);
          }
          for (; col < C4; col += stride) {
            // 为了不重复计算exp, 访问显存
            float4 b0 = in_row4[col];
            float4 v0 = make_float4(__expf(b0.x - maxval),
                                    __expf(b0.y - maxval),
                                    __expf(b0.z - maxval),
                                    __expf(b0.w - maxval));
            tile_sum_tmp += (v0.x + v0.y) + (v0.z + v0.w);
          }
          for (int col = (C4 << 2) + tid; col < tile_cols; col += blockDim.x){
            float v = __expf(in_row_tile[col] - maxval);
            tile_sum_tmp += v;
          }
        } else {
          for (; col + stride < C4; col += 2 * stride) {
            float4 a0 = smem4[col];
            float4 a1 = smem4[col + stride];
            float4 v0 = make_float4(__expf(a0.x - maxval),
                                    __expf(a0.y - maxval),
                                    __expf(a0.z - maxval),
                                    __expf(a0.w - maxval));
            float4 v1 = make_float4(__expf(a1.x - maxval),
                                    __expf(a1.y - maxval),
                                    __expf(a1.z - maxval),
                                    __expf(a1.w - maxval));
            smem4[col] = v0;
            smem4[col + stride] = v1;
            tile_sum_tmp += (v0.x + v0.y) + (v0.z + v0.w);
            tile_sum_tmp += (v1.x + v1.y) + (v1.z + v1.w);
          }
          for (; col < C4; col += stride) {
            float4 a0 = smem4[col];
            float4 v0 = make_float4(__expf(a0.x - maxval),
                                    __expf(a0.y - maxval),
                                    __expf(a0.z - maxval),
                                    __expf(a0.w - maxval));
            smem4[col] = v0;
            tile_sum_tmp += (v0.x + v0.y) + (v0.z + v0.w);
          }
          for (int col = (C4 << 2) + tid; col < tile_cols; col += blockDim.x){
            float v = __expf(smem[col] - maxval);
            smem[col] = v;
            tile_sum_tmp += v;
          }
        }
      } else {
        if constexpr (is_tile) {
          for (int col = tid; col < tile_cols; col += blockDim.x) {
            float v = __expf(in_row_tile[col] - maxval);
            tile_sum_tmp += v;;
          }
        } else {
          for (int col = tid; col < tile_cols; col += blockDim.x) {
            float v = __expf(smem[col] - maxval);
            smem[col] = v;
            tile_sum_tmp += v;
          }
        }
      }

      // 求得每个warp的和
      tile_sum_tmp = warpReduceSum(tile_sum_tmp);
      if (lane_id == 0) {
        smem_sum[warp_id] = tile_sum_tmp;
      }
      __syncthreads();

      // tid0求解当前block中的和
      // 此时求出的和是当前tile的和
      if (warp_id == 0) {
        tile_sum_tmp = (lane_id < warp_per_block) ? smem_sum[lane_id] : 0;
        tile_sum_tmp = warpReduceSum(tile_sum_tmp);
        if (lane_id == 0) {
          smem_tile_sum_scalar += tile_sum_tmp;
        }
      }
      __syncthreads();
    }
    sum = smem_tile_sum_scalar;
    float norm = __fdividef(1.0f, sum);
    #endif

    // 计算并输出
    #if 1
    for (int t = 0; t < tile_num; ++t) {
      int col = tid;
      int stride = blockDim.x;
      float* out_row_tile = out_row + t * tile_elems;
      int tile_cols = min(tile_elems, C - t*tile_elems);
      if (aligned16_out) {
        float4* smem4 = reinterpret_cast<float4*>(smem);
        float4* out_row4 = reinterpret_cast<float4*>(out_row_tile);
        int C4 = tile_cols >> 2;

        if constexpr (is_tile) {
          for (; col + stride < C4; col += 2 * stride) {
            float4 b0 = out_row4[col];
            float4 b1 = out_row4[col + stride];
            out_row4[col] = make_float4(__expf(b0.x - maxval) * norm,
                                        __expf(b0.y - maxval) * norm,
                                        __expf(b0.z - maxval) * norm,
                                        __expf(b0.w - maxval) * norm);
            out_row4[col + stride] = make_float4(__expf(b1.x - maxval) * norm,
                                                 __expf(b1.y - maxval) * norm,
                                                 __expf(b1.z - maxval) * norm,
                                                 __expf(b1.w - maxval) * norm);
          }
          for (; col < C4; col += stride) {
            // 为了不重复计算exp, 访问显存
            float4 b0 = out_row4[col];
            out_row4[col] = make_float4(__expf(b0.x - maxval) * norm,
                                        __expf(b0.y - maxval) * norm,
                                        __expf(b0.z - maxval) * norm,
                                        __expf(b0.w - maxval) * norm);
          }
          for (int col = (C4 << 2) + tid; col < tile_cols; col += blockDim.x){
            out_row_tile[col] = __expf(out_row_tile[col] - maxval) * norm;
          }
        } else {
          for (; col + stride < C4; col += 2 * stride) {
            float4 a0 = smem4[col];
            float4 a1 = smem4[col + stride];
            out_row4[col] = make_float4(a0.x * norm,
                                        a0.y * norm,
                                        a0.z * norm,
                                        a0.w * norm);
            out_row4[col + stride] = make_float4(a1.x * norm,
                                                 a1.y * norm,
                                                 a1.z * norm,
                                                 a1.w * norm);
          }
          for (; col < C4; col += stride) {
            float4 a0 = smem4[col];
            out_row4[col] = make_float4(a0.x * norm,
                                        a0.y * norm,
                                        a0.z * norm,
                                        a0.w * norm);
          }
          for (int col = (C4 << 2) + tid; col < tile_cols; col += blockDim.x){
            out_row_tile[col] = smem[col] * norm;
          }
        }
      } else {
        if constexpr (is_tile) {
          for (int col = tid; col < tile_cols; col += blockDim.x) {
            out_row_tile[col] = __expf(out_row_tile[col] - maxval) * norm;
          }
        } else {
          for (int col = tid; col < tile_cols; col += blockDim.x) {
            out_row_tile[col] = smem[col] * norm;
          }
        }
      }
      if constexpr (!is_tile) __syncthreads();
    }
    #endif
  }
}

int main() {
  int N = 4096;
  int C = 16384;

  size_t ele_num = N * C;
  float* in_h = (float*)malloc(ele_num * sizeof(float));
  float* out_h = (float*)malloc(ele_num * sizeof(float));
  float* out_res = (float*)malloc(ele_num * sizeof(float));

  float* in_d, *out_d;
  cudaMalloc((void**)&in_d, ele_num * sizeof(float));
  cudaMalloc((void**)&out_d, ele_num * sizeof(float));

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(-100.0f, 100.f);
  for (int i = 0; i < ele_num; ++i) {
    in_h[i] = dist_float(mt);
  }

  auto start_cpu = std::chrono::high_resolution_clock::now();
  safe_softmax_forward_CPU(out_h, in_h, N, C);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

  cudaDeviceProp prop{};
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);

  int optin = 0;
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  printf("最大 opt-in shared/blk: %d\n", optin);

  printf("GPU: %s\n", prop.name);
  // smemPerBlock = 49152 bytes
  printf("smemPerBlock      = %zu bytes\n", prop.sharedMemPerBlock);
  printf("smemPerBlockOptin = %zu bytes\n", prop.sharedMemPerBlockOptin);
  printf("smemPerSM         = %zu bytes\n", prop.sharedMemPerMultiprocessor);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int blockSize = 1024;
  int numBlocks = 512;

  cudaMemcpy(in_d, in_h, ele_num * sizeof(float), cudaMemcpyHostToDevice);

  //  smem中保存warp_per_block个maxval和sumval
  //  smem: 0 ~ warp_per_block-1: maxval
  //        warp_per_block ~ warp_per_block + warp_per_block - 1: sumval
  #if 0
    uint32_t smem_size =
        (blockSize * sizeof(float) + 512 - 1) / 512 * 512;
  #elif 0
    size_t smem_size = 2 * ((blockSize + 31) / 32) * sizeof(float);
  #elif 1
    // 动态与静态smem结合
    //  动态: C   静态: max[32], sum[32]
    auto align_up = [](size_t x, size_t a) {
      return (x + a - 1) / a * a;
    };

    int smem_size_max = 96 * 1024;
    size_t bytes = std::min<size_t>(smem_size_max, C * sizeof(float));
    size_t smem_size = align_up(bytes, 16);
    printf("smem_size: %ld\n", smem_size);
    cudaFuncSetAttribute(
        safe_softmax_forward_GPU_v4<true>,   // 或 <true>，看你要设置哪个实例
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size
    );
    cudaFuncSetAttribute(
        safe_softmax_forward_GPU_v4<false>,   // 或 <true>，看你要设置哪个实例
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size
    );
  #endif

  if (smem_size_max < (C * sizeof(float))) {
    printf("enter tile\n");
    for (int i = 0; i < 10; ++i) {
      safe_softmax_forward_GPU_v4<true><<<numBlocks, blockSize, smem_size>>>(
        out_d, in_d, N, C, smem_size);
    }

    // 立刻检查 launch 参数/配置错误（比如 invalid configuration）
    CUDA_CHECK(cudaPeekAtLastError());

    // 强制同步：如果 kernel 内部非法访存等运行时错误，这里会报出来
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    safe_softmax_forward_GPU_v4<true><<<numBlocks, blockSize, smem_size>>>(
      out_d, in_d, N, C, smem_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  } else {
    printf("not enter tile\n");
    for (int i = 0; i < 10; ++i) {
      safe_softmax_forward_GPU_v4<false><<<numBlocks, blockSize, smem_size>>>(
        out_d, in_d, N, C, smem_size);
    }

    // 立刻检查 launch 参数/配置错误（比如 invalid configuration）
    CUDA_CHECK(cudaPeekAtLastError());

    // 强制同步：如果 kernel 内部非法访存等运行时错误，这里会报出来
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    safe_softmax_forward_GPU_v4<false><<<numBlocks, blockSize, smem_size>>>(
      out_d, in_d, N, C, smem_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  }

  float gpu_time_ms = 0;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);

  cudaMemcpy(out_res, out_d, ele_num * sizeof(float), cudaMemcpyDeviceToHost);

  bool success = compare_results(out_h, out_res, N, C);
  std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

  // Print performance comparison
  std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
  std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
  std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x"
            << std::endl;


  free(in_h);
  free(out_h);
  free(out_res);
  cudaFree(in_d);
  cudaFree(out_d);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}