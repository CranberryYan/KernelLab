#include <cstdlib>
#include "op/add.h"
#include "./base/API_trace.h"
#include "kernels/kernels_interface.h"

namespace op {
static const bool apiTraceEnabled = (std::getenv("api_trace") != nullptr);

VecAddLayer::VecAddLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "Add") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status VecAddLayer::checkArgs() const {
  tensor::Tensor input1 = this->get_input(0);
  tensor::Tensor input2 = this->get_input(1);
  int32_t size = input1.size();
  base::Status status;
  status = check_tensor_with_dim(input1, device_type_, data_type_, size);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return status;
  }

  status = check_tensor_with_dim(input2, device_type_, data_type_, size);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the add layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, size);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the add layer.";
    return status;
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  return base::error::Success();
}

base::Status VecAddLayer::compute() {
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);

  para::add_para para;
  para.ele_num = static_cast<int32_t>(input1.size());
  para.thread_num = 1024;
  para.block_num = (para.ele_num + para.thread_num - 1) / para.thread_num;

  kernel::get_add_kernel(device_type_)(input1, input2, output, para,
                                       cuda_config_ ?
                                       cuda_config_->stream : nullptr);

  return base::error::Success();
}

base::Status VecAddLayer::forward() {
  if (apiTraceEnabled) {
    api_trace::API_trace trace("VecAddLayer::forward()");
    trace.set_tensor("lhs", this->inputs_[0]);
    trace.set_tensor("rhs", this->inputs_[1]);
    trace.set_tensor("out", this->outputs_[0]);

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
