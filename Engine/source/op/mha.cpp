#include "op/mha.h"
#include "./base/API_trace.h"
#include "kernels/cpu/mha_kernel.h"
#include "kernels/kernels_interface.h"

namespace op {
  static const bool apiTraceEnabled = (std::getenv("api_trace") != nullptr);

MHA::MHA(base::DeviceType device_type, int32_t layer_index,
         int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
         int32_t head_num, int32_t head_size) :
  Layer(device_type, LayerType::kLayerMHA, "MultiHead"),
  layer_index_(layer_index), kv_mul_(kv_mul), kv_dim_(kv_dim),
  seq_len_(seq_len), head_num_(head_num), head_size_(head_size) {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status MHA::checkArgs() const {
  base::Status status;
  const int32_t input_tensor_num = 4;
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    // mha score tensor
    status = check_tensor(get_input(i), device_type_, data_type_);
    if (!status) {
      LOG(ERROR) << "The input tensor " <<
        std::to_string(i) << " error in the matmul layer.";
      return status;
    }
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  return check_tensor(get_output(0), device_type_, data_type_);
}

base::Status MHA::compute() {
  tensor::Tensor& mha_out = this->get_output(0);
  const tensor::Tensor& query_tensor = this->get_input(0);
  const tensor::Tensor& score_tensor = this->get_input(1);
  const tensor::Tensor& key_cache_tensor = this->get_input(2);
  const tensor::Tensor& value_cache_tensor = this->get_input(3);

  kernel::get_mha_kernel(device_type_)(pos_, head_num_, layer_index_,
                                       seq_len_, kv_dim_, kv_mul_,
                                       head_size_, mha_out,
                                       query_tensor,
                                       score_tensor,
                                       key_cache_tensor,
                                       value_cache_tensor,
                                       device_type_,
                                       cuda_config_ ?
                                       cuda_config_.get() :
                                       nullptr);

  return base::error::Success();
}

base::Status MHA::forward() {
  if (apiTraceEnabled) {
    api_trace::API_trace trace("MHA::forward()");
    trace.set_tensor("query_tensor", this->get_input(0));
    trace.set_tensor("score_tensor", this->get_input(1));
    trace.set_tensor("key_cache_tensor", this->get_input(2));
    trace.set_tensor("value_cache_tensor", this->get_input(3));
    trace.set_tensor("output", this->get_output(0));

    trace.print_tensor();
  }

  auto status = checkArgs();
  if (!status) {
    return status;
  }

  status = this->compute();
  if (!status) {
    return status;
  }

  return base::error::Success();
}

void MHA::set_pos(int32_t pos) { this->pos_ = pos; }

void MHA::set_layer_idx(int32_t layer_idx) { this->layer_index_ = layer_idx; }
} // namespace op
