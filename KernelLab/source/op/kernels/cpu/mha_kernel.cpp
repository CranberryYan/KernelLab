#include "mha_kernel.h"
#include "../kernels_interface.h"

namespace kernel {
#define DEBUG 1

void mha_kernel(int32_t pos, int32_t head_num,
                int32_t layer_index, int32_t seq_len,
                int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor,
                const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor,
                const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config) {
#if 0
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));

  std::shared_ptr<base::DeviceAllocator> allocator;
  if (device_type == base::DeviceType::kDeviceCPU) {
    allocator = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    allocator = base::CUDADeviceAllocatorFactory::get_instance();
  }

#if DEBUG
std::vector<int32_t> q_dim = query_tensor.dims();
std::vector<int32_t> score_dim = score_tensor.dims();
std::vector<int32_t> key_cache_dim = key_cache_tensor.dims();
std::vector<int32_t> value_cache_dim = key_cache_tensor.dims();
std::vector<int32_t> mha_out_dim = mha_out.dims();
#endif

  for (int h = 0; h < head_num; ++h) {
    float* score_head_addr =
      const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
    float* query_head_addr =
      const_cast<float*>(query_tensor.ptr<float>() + h * head_size);

    tensor::Tensor query_mat(base::DataType::kDataTypeFp32, 1, head_size,
                             false, nullptr, query_head_addr);
    query_mat.set_device_type(device_type);

    for (int t = 0; t <= pos; ++t) {
      // kv_mul: MHA, 一组kv负责一个q, 固定为1
      int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;
      const float* key_head_addr =
        key_cache_tensor.ptr<float>() + layer_offset + cache_offset;

      tensor::Tensor key_mat(base::DataType::kDataTypeFp32, 1, head_size,
        false, nullptr, const_cast<float*>(key_head_addr));

      tensor::Tensor score_mat(base::DataType::kDataTypeFp32, 1,
        false, nullptr, score_head_addr + t);

      key_mat.set_device_type(device_type);
      score_mat.set_device_type(device_type);
      get_matmul_kernel(device_type)(query_mat, key_mat, score_mat,
                                     scale, config);
    }

    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, pos + 1,
                                     false, nullptr, score_head_addr);
    score_head_tensor.set_device_type(device_type);
    get_softmax_kernel(device_type)(score_head_tensor,
                                    config ? config->stream : nullptr);

    float* output_head_ptr = mha_out.ptr<float>() + h * head_size;
    allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                           config ? config->stream : nullptr, false);
    tensor::Tensor output_tensor(base::DataType::kDataTypeFp32, head_size,
                                 false, nullptr, output_head_ptr);
    output_tensor.set_device_type(device_type);

    int32_t cache_offset = (h / kv_mul) * head_size;
    float* value_head_addr =
      const_cast<float*>(value_cache_tensor.ptr<float>()) +
        layer_offset + cache_offset;
    tensor::Tensor value_tensor(base::DataType::kDataTypeFp32, head_size,
                                false, nullptr, value_head_addr);

    // for (int i = 0; i <= pos; ++i) {
    //   arma::fvec value_vec(const_cast<float*>(value.ptr<float>()) +
    //                        i * stride, value.size(), false, true);
    //   output_vec += scale_vec[i] * value_vec;
    // }
    //  scale_vec: 每个pos(token)对应一个scale
    //  output_vec: [1, head_size]
    //  综上:
    //    当前pos的输入Q, 会和所有的K进行计算, K: [head_num, pos, head_size]
    //    总计计算pos-1 + 1次向量乘法 -> 得出pos个scale -> scale_tensor: [1, pos]
    //    pos-1: 历史的K经过linear得出的K
    //    1: 当前的K经过linear得出的K
    //    scale_tensor: [1, pos] -> softmax -> scale_tensor: [1, pos]
    //    V: [head_num, pos, head_size]
    //    pos-1: 历史的V经过linear得出的V   1: 当前的V经过linear得出的V 
    //    output: [head_num, pos, head_size]
    //    将scale_tensor中的pos个scale, 对应相乘 output_vec += scale_vec[i] * value_vec;
    //  循环以上head_num次
    get_scale_sum_kernel(device_type)(value_tensor,
                                      score_head_tensor,
                                      output_tensor,
                                      pos, head_size, kv_dim,
                                      config ? config->stream : nullptr);
  }
#endif
}
} // namespace kernel
