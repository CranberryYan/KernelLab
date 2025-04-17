#include "op/layer.h"
#include "op/embedding.h"
#include "./base/API_trace.h"
#include "kernels/kernels_interface.h"

namespace op {
static const bool apiTraceEnabled = (std::getenv("api_trace") != nullptr);

EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim,
                               int32_t seq_len, int32_t vocab_size) :
  dim_(dim), seq_len_(seq_len), vocab_size_(vocab_size),
  LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding") {

  reset_input_size(2);
  reset_weight_size(1);
  reset_output_size(1);
}

base::Status EmbeddingLayer::checkArgs() const {
  const tensor::Tensor &input_tensor = get_input(0);
  const int32_t &token_size = get_input(1).size();
  if (token_size > input_tensor.size()) {
    return base::error::InvalidArgument(
      "The number of input tensor is greater than seq len.");
  }

  base::Status status = check_tensor_with_dim(input_tensor,
                                              base::DeviceType::kDeviceCPU,
                                              base::DataType::kDataTypeInt32,
                                              token_size);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_weight(0),
                                 device_type_, data_type_,
                                 vocab_size_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0),
                                 device_type_, data_type_, token_size, dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding layer.";
    return status;
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  return base::error::Success();
}

base::Status EmbeddingLayer::compute() {
  kernel::get_embedding_kernel(device_type_)(get_input(0),
                                             get_weight(0),
                                             get_output(0),
                                             vocab_size_,
                                             cuda_config_ ?
                                             cuda_config_->stream : nullptr);

  return base::error::Success();
}

base::Status EmbeddingLayer::forward() {
  if (apiTraceEnabled) {
    api_trace::API_trace trace("EmbeddingLayer::forward()");
    trace.set_tensor("input", this->get_input(0));
    trace.set_tensor("input_token_num", this->get_input(1));
    trace.set_tensor("weight", this->get_weight(0));
    trace.set_tensor("output", this->get_output(0));

    trace.print_tensor();
  }

  base::Status status = checkArgs();
  if (!status) {
    return status;
  }

  status = this->compute();
  if (!status) {
    return status;
  }

  return base::StatusCode::kSuccess;
}
} // namespace op
