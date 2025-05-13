#include "rope_kernel.h"

namespace kernel
{
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len,
                            float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq = 1.0f /std::pow(
        10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fsi = sinf(val);
      float fco = cosf(val);
      *(sin_cache + pos * head_size + head_dim) = fsi;
      *(cos_cache + pos * head_size + head_dim) = fco;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size,
                     const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache,
                     const tensor::Tensor& cos_cache,
                     void* stream) {
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int i = 0; i < dim; i += 2) {
    int head_dim = i % head_size;
    float fsi = *(sin_cache.ptr<float>() + pos * head_size + head_dim);
    float fco = *(cos_cache.ptr<float>() + pos * head_size + head_dim);

    int rotn = i < kv_dim ? 2 : 1;
    for (int v = 0; v < rotn; ++v) {
      float* vec = 
        const_cast<float*>(v == 0 ? input_q.ptr<float>()
                                  : input_k.ptr<float>());
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i] = v0 * fco - v1 * fsi;
      vec[i + 1] = v0 * fsi + v1 * fco;
    }
  }
}
} // namespace kernel
