#include "op/layer.h"
#include "op/rmsnorm.h"
#include "./base/API_trace.h"
#include "kernels/kernels_interface.h"

namespace op {
  static const bool apiTraceEnabled = (std::getenv("api_trace") != nullptr);

RMSNormLayer::RMSNormLayer(base::DeviceType device_type, int32_t dim)
    : LayerParam(device_type, LayerType::kLayerRMSNorm, false, "RMSNorm"),
    dim_(dim) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
}

base::Status RMSNormLayer::checkArgs() const {
  base::Status status = check_tensor_with_dim(get_input(0),
                                              device_type_,
                                              data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the rmsnorm layer.";
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the rmsnorm layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the rmsnorm layer.";
    return status;
  }

  return base::error::Success();
}

base::Status RMSNormLayer::compute() {
  tensor::Tensor input = this->get_input(0);
  tensor::Tensor weight = this->get_weight(0);
  tensor::Tensor output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_rmsnorm_kernel(device_type_)(input, weight,
                                           output,
                                           cuda_config_ ?
                                           cuda_config_->stream : nullptr);

  return base::error::Success();
}

base::Status RMSNormLayer::forward() {
  if (apiTraceEnabled) {
    api_trace::API_trace trace("RMSNormLayer::forward()");
    trace.set_tensor("input", this->get_input(0));
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

  return base::error::Success();
}
} // namespace op
