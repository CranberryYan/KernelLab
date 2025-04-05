#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include "tensor/tensor.h"
#include "base/cuda_config.h"

namespace kernel {
constexpr static int thread_num = 256;










__global__ void multi_head_attention_kernel(
  int32_t pos, int32_t seq_len, float* query,
  float* score_ptr, float* output,
  float* key_cache, float* value_cache,
  int32_t kv_dim, int32_t kv_mul,
  int32_t head_num, int32_t head_size, int32_t layer_offset) {
  





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
