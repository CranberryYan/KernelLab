#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "model/model.h"

namespace model
{
std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(
  int32_t layer_idx, int32_t token_pos) const {
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(
        get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(
        get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32,
                     config_->kv_dim_,
                     false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32,
                     config_->kv_dim_,
                     false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);

  return {key, val};
}

} // namespace model
