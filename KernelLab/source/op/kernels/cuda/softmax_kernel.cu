#include <cuda_runtime_api.h>
#include "math_utils.cuh"
#include "base/para.h"
#include "softmax_kernel.cuh"

namespace kernel {
// 返回当前这个warp的max
__inline__ __device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__inline__ __device__ float block_reduce_max(float val) {
  __shared__ float shared_max[32]; // Ampere最多分配1024个thread -> 32个warp
  uint32_t lane_id = threadIdx.x % 32;
  uint32_t wid = threadIdx.x / 32;

  for (int i = 0; i < 32; ++i) {
    shared_max[i] = -INFINITY;
  }
  __syncthreads();

  val = warp_reduce_max(val); // val: 当前这个wid中的最大值
  if (lane_id == 0) {
    shared_max[wid] = val;
  }
  __syncthreads();

  // 赋值, 把shared_max的值赋给前32个thread(第一个warp), 让其规约
  // shared_max: 
  val = (threadIdx.x < 32) ? shared_max[threadIdx.x] : -INFINITY;

  if (wid == 0) {
    val = warp_reduce_max(val); // 拿到当前block的max
  }

  return (threadIdx.x == 0) ? val : -INFINITY;
}

// 返回当前这个warp的所有值的sum
__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
  __shared__ float shared_sum[32];
  uint32_t lane_id = threadIdx.x % 32;
  uint32_t wid = threadIdx.x / 32;

  for (int i = 0; i < 32; ++i) {
    shared_sum[i] = 0.0f;
  }
  __syncthreads();

  val = warp_reduce_sum(val); // val: 当前这个wid中的所有元素的sum
  if (lane_id == 0) {
    shared_sum[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < 32) ? shared_sum[threadIdx.x] : 0;

  val = warp_reduce_sum(val);

  return val;
}

__global__ void naive_softmax_kernel_v0(const float* input, float* output,
                                        const uint32_t rows,
                                        const uint32_t cols) {
  uint32_t bid = blockIdx.x;
  uint32_t tid = threadIdx.x;

  // Naive版本, 仅需要求出sum
  __shared__ float sum;
  extern __shared__ float shared_cols[];
  for (int r = bid; r < rows; r += gridDim.x) {
    if (tid == 0) {
      sum = 0;
    }
    for (int c = tid; c < cols; c += blockDim.x) {
      shared_cols[c] = expf(input[r * cols + c]);
    }
    __syncthreads();

    if (tid == 0) {
      for (int c = 0; c < cols; ++c) {
        sum += shared_cols[c];
      }
    }
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
      output[r * cols + c] = shared_cols[c] / sum;
    }
    __syncthreads();
  }
}

__global__ void safe_softmax_kernel_v0(const float* input, float* output,
                                       const uint32_t rows,
                                       const uint32_t cols) {
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;

  __shared__ float sum;
  __shared__ float max_val;
  extern __shared__ float shared_cols[];
  for (int r = bid; r < rows; r += gridDim.x) {
    if (tid == 0) {
      sum = 0;
      max_val = -INFINITY;
    }
    for (int c = tid; c < cols; c += blockDim.x) {
      shared_cols[c] = input[r * cols + c];
    }
    __syncthreads();

    for (int c = 0; c < cols; ++c) {
      max_val = max(max_val, shared_cols[c]);
    }
    __syncthreads();

    for (int c = 0; c < cols; ++c) {
      sum += expf(shared_cols[c] - max_val);
    }
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
      output[r * cols + c] = expf(shared_cols[c] - max_val) / sum;
    }
  }
}

// bug: 一整列全部放入smem, 如果当前列过大, smem空间不够
__global__ void safe_softmax_kernel_v1(const float* input, float* output,
                                       const uint32_t rows,
                                       const uint32_t cols) {
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;

  __shared__ float sum_block;
  __shared__ float max_block;
  extern __shared__ float shared_cols[];

  uint32_t cols_4 = cols / 4;
  for (int r = bid; r < rows; r += gridDim.x) {
    for (int c = tid; c < cols_4; c += blockDim.x) {
      shared_cols[c * 4 + 0] = input[r * cols + c * 4 + 0];
      shared_cols[c * 4 + 1] = input[r * cols + c * 4 + 1];
      shared_cols[c * 4 + 2] = input[r * cols + c * 4 + 2];
      shared_cols[c * 4 + 3] = input[r * cols + c * 4 + 3];
    }
    for (int c = cols_4 * 4 + tid; c < cols; c += blockDim.x) {
      shared_cols[c] = input[r * cols + c];
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int c = tid; c < cols; c += blockDim.x) {
      local_max = max(local_max, shared_cols[c]);
    }
    float max_tmp = block_reduce_max(local_max);
    if (tid == 0) {
      max_block = max_tmp;
    }
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
      shared_cols[c] = expf(shared_cols[c] - max_block);
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
      local_sum += shared_cols[c];   // 注意此时 shared_cols 已经是 exp 后的值
    }
    float sum_tmp = block_reduce_sum(local_sum);
    if (tid == 0) {
      sum_block = sum_tmp;
    }
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
      output[r * cols + c] = shared_cols[c] / sum_block;
    }
    __syncthreads();
  }
}

__inline__ __device__ float warpReduceMax(float val) {
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

__inline__ __device__ float warpReduceSum(float val) {
  int lane = threadIdx.x & 31;
  for (int stride = 16; stride >= 1; stride >>= 1) {
    unsigned mask = __activemask();
    float other = __shfl_down_sync(mask, val, stride);
    if (lane < stride) val += other;
  }
  return val; // lane0 是整个 warp max
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
__global__ void safe_softmax_kernel_v2(const float* input, float* output,
                                       const uint32_t rows,
                                       const uint32_t cols,
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

  for (int row = bid; row < rows; row += gridDim.x) {
    float maxval = -INFINITY;
    float sum = 0.0f;
    if (tid == 0) {
      smem_tile_max_scalar = -INFINITY; smem_tile_sum_scalar = 0.0f;
    }
    __syncthreads();

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;
    bool aligned16_in = (((uintptr_t)in_row) & 0x000F) == 0;
    bool aligned16_out = (((uintptr_t)out_row) & 0x000F) == 0;

    const int tile_size = smem_size;
    const int tile_elems = tile_size / sizeof(float);
    int tile_num = (cols + tile_elems - 1) / tile_elems;
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
      int tile_cols = min(tile_elems, cols - t*tile_elems);

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
      int tile_cols = min(tile_elems, cols - t*tile_elems);
      if (aligned16_in && aligned16_out) {
        const float4* in_row4 = reinterpret_cast<const float4*>(in_row_tile);
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
      int tile_cols = min(tile_elems, cols - t*tile_elems);
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

__global__ void online_softmax_kernel_v0(const float* input, float* output,
                                         const uint32_t rows,
                                         const uint32_t cols) {
  // 一个thread负责一行
  //  FlashAttention中, 行比较大, 列比较小
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t r = gid;
  if (r >= rows) {
    return;
  }

  float sum = 0;
  float max_tmp = -INFINITY;
  float pre_max_tmp = 0;
  for (int c = 0; c < cols; ++c) {
    uint32_t input_offset = r * cols + c;
    max_tmp = max(max_tmp, input[input_offset]);
    sum =
      sum * expf(pre_max_tmp - max_tmp) +
        expf(input[input_offset] - max_tmp);
    pre_max_tmp = max_tmp;
  }

  for (int c = 0; c < cols; ++c) {
    uint32_t input_offset = r * cols + c;
    output[input_offset] = expf(input[input_offset] - max_tmp) / sum;
  }
}

__global__ void online_softmax_kernel_v1(const float* input, float* output,
                                         const uint32_t rows,
                                         const uint32_t cols) {
  // 一个blokc负责n行, 最多1024个block, stride: gridDim.x
  //  一个block中的thread负责n个元素, 最多128个thread, stride: blockDim.x
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  if (bid >= rows || tid >= cols) {
    return;
  }

  extern __shared__ float shared_cols[];
  __shared__ float sum;
  __shared__ float max_tmp;
  __shared__ float pre_max_tmp;
  for (int b = bid; b < rows; b += gridDim.x) {
    if (tid == 0) {
      sum = 0;
      max_tmp = -INFINITY;
      pre_max_tmp = 0;
    }
    for (int c = tid; c < cols; c += blockDim.x) {
      shared_cols[c] = input[b * cols + c];
    }
    __syncthreads();
    if (tid == 0) {
      for (int c = 0; c < cols; ++c) {
        max_tmp = max(max_tmp, shared_cols[c]);
        sum =
          sum * expf(pre_max_tmp - max_tmp) +
            expf(shared_cols[c] - max_tmp);
        pre_max_tmp = max_tmp;
      }
    }
    __syncthreads();

    for (int c = tid; c < cols; c += blockDim.x) {
      output[b * cols + c] = expf(shared_cols[c] - max_tmp) / sum;
    }
    __syncthreads();
  }
}

void softmax_kernel_cu(const tensor::Tensor& input,
                        tensor::Tensor& output,
                        para::softmax_para para,
                        void* stream) {
  uint32_t rows = para.input_rows;
  uint32_t cols = para.input_cols;

  uint32_t thread_num = cols < 1024 ? cols : 1024;
  uint32_t block_num = rows < 512 ? rows : 512;

  dim3 block(thread_num);
  dim3 grid(block_num);

  if (para.op_type == para::SoftmaxOpType::Naive) {
    size_t smem_size = math_cu::AlignUp<uint32_t>(cols * sizeof(float), 512);
    if (stream) {
      cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
      naive_softmax_kernel_v0<<<grid, block, smem_size, stream_>>>(
        input.ptr<float>(), output.ptr<float>(), rows, cols);
    } else {
      naive_softmax_kernel_v0<<<grid, block, smem_size>>>(
        input.ptr<float>(), output.ptr<float>(), rows, cols);
    }
  } else if (para.op_type == para::SoftmaxOpType::Safe) {
    int smem_size_max = 96 * 1024;
    size_t bytes = std::min<size_t>(smem_size_max, cols * sizeof(float));
    size_t smem_size = math_cu::AlignUp<size_t>(bytes, 16);
    printf("smem_size: %ld\n", smem_size);
    cudaFuncSetAttribute(
        safe_softmax_kernel_v2<true>,   // 或 <true>，看你要设置哪个实例
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size
    );
    cudaFuncSetAttribute(
        safe_softmax_kernel_v2<false>,   // 或 <true>，看你要设置哪个实例
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size
    );

    if (smem_size_max < (cols * sizeof(float))) {
      printf("enter tile\n");
      if (stream) {
        cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
        safe_softmax_kernel_v2<true><<<grid, block, smem_size, stream_>>>(
          input.ptr<float>(), output.ptr<float>(), rows, cols, smem_size);
      } else {
        safe_softmax_kernel_v2<true><<<grid, block, smem_size>>>(
          input.ptr<float>(), output.ptr<float>(), rows, cols, smem_size);
      }
    } else {
      printf("enter no_tile\n");
      if (stream) {
        cudaStream_t stream_ = static_cast<CUstream_st*>(stream);

        safe_softmax_kernel_v2<false><<<grid, block, smem_size, stream_>>>(
          input.ptr<float>(), output.ptr<float>(), rows, cols, smem_size);
      } else {
        safe_softmax_kernel_v2<false><<<grid, block, smem_size>>>(
          input.ptr<float>(), output.ptr<float>(), rows, cols, smem_size);
      }
    }
  } else if (para.op_type == para::SoftmaxOpType::Online) {
    size_t smem_size = math_cu::AlignUp<uint32_t>(cols * sizeof(float), 512);
    if (stream) {
      cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
      online_softmax_kernel_v1<<<grid, block, smem_size, stream_>>>(
        input.ptr<float>(), output.ptr<float>(), rows, cols);
    } else {
      online_softmax_kernel_v1<<<grid, block, smem_size>>>(
        input.ptr<float>(), output.ptr<float>(), rows, cols);
    }
  } else {
    printf("ERROR: Unknown SoftmaxOpType\n");
  }
}
} // namespace kernel
