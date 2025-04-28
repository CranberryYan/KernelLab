#include <cstdlib>
#include "op/index_add.h"
#include "./base/API_trace.h"
#include "kernels/kernels_interface.h"

namespace op {
static const bool apiTraceEnabled = (std::getenv("api_trace") != nullptr);

IndexAddLayer::IndexAddLayer(base::DeviceType device_type)
  : Layer(device_type, LayerType::kLayerIndexAdd, "IndexAdd") {
  reset_input_size(3);
  reset_output_size(1);
}

base::Status IndexAddLayer::checkArgs() const {
  tensor::Tensor input = this->get_input(0);
  tensor::Tensor index = this->get_input(1);
  tensor::Tensor source = this->get_input(2);
  tensor::Tensor output = this->get_output(0);

  if (input.get_buffer() != output.get_buffer()) {
    LOG(ERROR) << "The input tensor output tensor is not the same buffer.";
    return base::error::InvalidArgument(
      "The input tensor output tensor is not the same buffer.");
  }

  // index: 一维向量, 值域: (-target_rows, target_rows)
  // source: shape: [index.size(), self_cols]
  if (index.dims_size() != 1) {
    LOG(ERROR) << "The index tensor is invaild.";
    return base::error::InvalidArgument("The index tensor is invaild.");
  }

  uint32_t target_rows = input.get_dim(0);
  uint32_t target_cols = input.get_dim(1);
  int32_t low = -target_rows;
  int32_t high = target_rows;
  for (int i = 0; i < index.size(); ++i) {
    if (index.at<int32_t>(i) < low && index.at<int32_t>(i) > high) {
    LOG(ERROR) << "The index val is invaild.";
    return base::error::InvalidArgument("The index values is invaild.");
    }
  }

  if (source.get_dim(0) != index.size() && source.get_dim(0) != target_cols) {
    LOG(ERROR) << "The source tensor is invaild.";
    return base::error::InvalidArgument("The source tensor is invaild.");
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  return base::error::Success();
}

base::Status IndexAddLayer::compute() {
  tensor::Tensor input = this->get_input(0);
  tensor::Tensor index = this->get_input(1);
  tensor::Tensor source = this->get_input(2);
  tensor::Tensor output = this->get_output(0);

  para::index_add_para para;
  para.bpe = sizeof(float);
  para.enflame_device = this->enflame_device;
  para.target_dims = input.dims();
  para.index_dims = index.dims();
  para.source_dims = source.dims();
  para.target_rows = para.target_dims[0];
  para.target_cols = para.target_dims[1];
  para.index_nums = para.index_dims[0];
  para.source_rows = para.source_dims[0];
  para.source_cols = para.source_dims[1];

  kernel::get_index_add_kernel(device_type_)(input, index, source, output, para,
                                             cuda_config_ ?
                                             cuda_config_->stream : nullptr);

  return base::error::Success();
}

base::Status IndexAddLayer::forward() {
  if (apiTraceEnabled) {
    api_trace::API_trace trace("IndexAddLayer::forward()");
    trace.set_tensor("target", this->inputs_[0]);
    trace.set_tensor("index", this->inputs_[1]);
    trace.set_tensor("source", this->inputs_[2]);

    trace.print_tensor();
  }

  base::Status status = this->checkArgs();
  if (!status) {
    return status;
  }

  status = this->compute();
  if (!status) {
    return status;
  }

  return base::error::Success();
}
} // namespace op
