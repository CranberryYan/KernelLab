#include <cstdlib>
#include "op/reduce.h"
#include "./base/API_trace.h"
#include "kernels/kernels_interface.h"

namespace op {
static const bool apiTraceEnabled = (std::getenv("api_trace") != nullptr);

ReduceLayer::ReduceLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerReduce, "Reduce") {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status ReduceLayer::checkArgs() const {
  tensor::Tensor input = this->get_input(0);
  tensor::Tensor output = this->get_output(0);
  int32_t size = input.size();
  int32_t after_reduce_size = output.size();
  base::Status status;
  status = check_tensor_with_dim(input, device_type_, data_type_, size);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the reduce layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0),
    device_type_, data_type_, after_reduce_size);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the reduce layer.";
    return status;
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  return base::error::Success();
}

base::Status ReduceLayer::compute() {
  auto input = this->get_input(0);
  auto output = this->get_output(0);

  para::reduce_para para;
  para.ele_num = static_cast<uint32_t>(input.size());
  para.after_reduce_num = static_cast<uint32_t>(output.size());
  para.block_num = para.after_reduce_num;
  para.thread_num = (para.ele_num + para.block_num - 1) / para.block_num;

  kernel::get_reduce_kernel(device_type_)(input, output, para,
                                          cuda_config_ ?
                                          cuda_config_->stream : nullptr);

  return base::error::Success();
}

base::Status ReduceLayer::forward() {
  if (apiTraceEnabled) {
    api_trace::API_trace trace("ReduceLayer::forward()");
    trace.set_tensor("input", this->inputs_[0]);
    trace.set_tensor("output", this->outputs_[0]);

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
