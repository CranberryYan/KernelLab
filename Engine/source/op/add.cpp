#include <cstdlib>
#include "op/add.h"
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
  // printf("apiTraceEnabled: %d\n", apiTraceEnabled);
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

void print_tensor(std::string name, tensor::Tensor ten) {
  printf("%s\n", name.c_str());
  printf("dataType: %d\n", static_cast<int>(ten.device_type()));
  for (int dim = 0; dim < ten.dims().size(); ++dim) {
    printf("%s[%d]: %d\n", name.c_str(), dim, ten.get_dim(dim));
  }
}

base::Status VecAddLayer::forward() {
  if (apiTraceEnabled) {
    printf("API_TRACE:\n");
    printf("API_NAME: VecAddLayer::forward\n");
    print_tensor("VecAddLayer.input1", this->get_input(0));
  }

  auto status = this->checkArgs();
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
