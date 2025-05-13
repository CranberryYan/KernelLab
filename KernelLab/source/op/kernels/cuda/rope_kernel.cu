#include "rope_kernel.cuh"

namespace kernel {
__global__ void sin_cos_calc(int head_size, int max_seq_len,
                             float* sin_cache, float* cos_cache) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int head_dim = gid % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow
      (10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fsi = sinf(val);
    float fco = cosf(val);
    *(sin_cache + pos * head_size + head_dim) = fsi;
    *(cos_cache + pos * head_size + head_dim) = fco;
  }
}

__device__ void rope_calc(float fco, float fsi, float* vec, int32_t idx) {
  float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
  float2 vec_value = *vec_ptr;
  *vec_ptr =
    make_float2(vec_value.x * fco - vec_value.y * fsi,
                vec_value.x * fsi + vec_value.y * fco);
}

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q,
                                    const float* input_k,
                                    const float* sin_cache,
                                    const float* cos_cache) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = gid * 2;
  if (idx >= dim) {
    return;
  }

  int head_dim = idx % head_size;
  float fsi = *(sin_cache + pos * head_size + head_dim);
  float fco = *(cos_cache + pos * head_size + head_dim);

  rope_calc(fco, fsi, const_cast<float*>(input_q), idx);
  if (idx >= kv_dim) {
    return;
  }
  rope_calc(fco, fsi, const_cast<float*>(input_k), idx);
}

void sin_cos_cache_calc_cu(int head_size, int max_seq_len,
                           const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache,
                           cudaStream_t stream) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int threads = head_size;
  if (stream) {
    sin_cos_calc<<<1, threads, 0, stream>>>(
      head_size, max_seq_len,
      const_cast<float*>(sin_cache.ptr<float>()),
      const_cast<float*>(cos_cache.ptr<float>()));
  } else {
    sin_cos_calc<<<1, threads>>>(head_size, max_seq_len,
                                 const_cast<float*>(sin_cache.ptr<float>()),
                                 const_cast<float*>(cos_cache.ptr<float>()));
  }
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size,
                    const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k,
                    const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache,
                    const tensor::Tensor& cos_cache, void* stream) {
  const int32_t pos = *input_pos.ptr<int32_t>(0);
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
      pos, dim, kv_dim, head_size,
      input_q.ptr<float>(), input_k.ptr<float>(),
      sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_cu_fp32<<<blocks, threads>>>(
      pos, dim, kv_dim, head_size,
      input_q.ptr<float>(), input_k.ptr<float>(),
      sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}
} // namespace kernel
