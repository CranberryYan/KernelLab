#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include "tensor/tensor.h"
#include "base/cuda_config.h"

namespace kernel {
constexpr static int thread_num = 256;
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // scale_tensor: [1, head_num, 1, seq_len]
  // 需要找到head_num个max
  // 找到每个block中的max
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();

  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

__global__ void multi_head_attention_kernel(
  int32_t pos, int32_t seq_len, float* query,
  float* score_ptr, float* output,
  float* key_cache, float* value_cache,
  int32_t kv_dim, int32_t kv_mul,
  int32_t head_num, int32_t head_size, int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  // q: [1, head_num, 1, head_size]
  // k: [1, head_num, seq_len, head_size]
  // score: [1, head_num, seq_len, 1]
  float scale = 1.f / sqrtf(head_size);
  float* query_head = query + head * head_size;
  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;

  // 每个token(thread), q * k^T -> scale(标量)
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + head_offset + t * kv_dim;
    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_head_float4 = *reinterpret_cast<float4*>(key_head + i);
      float4 query_head_float4 = *reinterpret_cast<float4*>(query_head + i);
      score += key_head_float4.x * query_head_float4.x;
      score += key_head_float4.y * query_head_float4.y;
      score += key_head_float4.z * query_head_float4.z;
      score += key_head_float4.w * query_head_float4.w;
    }

    // scale_tensor: [1, head_num, 1, seq_len] -> softmax ->
    //  score_tensor: [1, head_num, 1, seq_len]
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + head_offset + t * kv_dim;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num,
                   int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                   tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor,
                   const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor,
                   const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  multi_head_attention_kernel<<<head_num, thread_num, 0, stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
}
} // namespace kernel
