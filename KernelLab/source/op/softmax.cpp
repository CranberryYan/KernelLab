#include "op/softmax.h"
#include "./base/API_trace.h"
#include "kernels/kernels_interface.h"

namespace op {
static const bool apiTraceEnabled = (std::getenv("api_trace") != nullptr);

SoftmaxLayer::SoftmaxLayer(base::DeviceType device_type) :
  Layer(device_type, LayerType::kLayerSoftmax, "Softmax") {
    reset_input_size(1);
    reset_output_size(1);
}

base::Status SoftmaxLayer::checkArgs() const {
  base::Status status;
  tensor::Tensor input = this->get_input(0);
  tensor::Tensor output = this->get_output(0);

  uint32_t input_ele_num = input.size();
  uint32_t output_ele_num = output.size();

  if (input_ele_num != output_ele_num) {
    return base::error::InvalidArgument("The tensor has a wrong ele num.");
  }

  status = check_tensor_with_dim(input, device_type_,
                                 data_type_,
                                 input.get_dim(0),
                                 input.get_dim(1));
  if (!status) {
    LOG(ERROR) << "The input tensor error in the softmax layer.";
    return status;
  }

  status = check_tensor_with_dim(output, device_type_,
                                 data_type_,
                                 output.get_dim(0),
                                 output.get_dim(1));
  if (!status) {
    LOG(ERROR) << "The output tensor error in the softmax layer.";
    return status;
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  return base::error::Success();
}

base::Status SoftmaxLayer::compute() {
  auto input = this->get_input(0);
  auto output = this->get_output(0);

  para::softmax_para para;
  para.op_type = this->op_type;
  para.input_dims = input.dims();
  para.output_dims = output.dims();
  para.input_rows = para.input_dims[0];
  para.input_cols = para.input_dims[1];

  kernel::get_softmax_kernel(device_type_)(input, output, para,
                                           cuda_config_ ?
                                           cuda_config_->stream : nullptr);

  return base::error::Success();
}

base::Status SoftmaxLayer::forward() {
  if (apiTraceEnabled) {
    api_trace::API_trace trace("SoftmaxLayer::forward()");
    trace.set_tensor("input", this->inputs_[0]);
    trace.set_tensor("output", this->outputs_[0]);
    trace.print_softmax_type(op_type);

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
