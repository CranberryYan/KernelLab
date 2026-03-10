#include "op/histogram.h"
#include "./base/API_trace.h"
#include "kernels/kernels_interface.h"

namespace op {
static const bool apiTraceEnabled = (std::getenv("api_trace") != nullptr);

HistogramLayer::HistogramLayer(int32_t low_, int32_t high_,
                               base::DeviceType device_type) :
  Layer(device_type,
        LayerType::kLayerHistogram,
        base::DataType::kDataTypeInt32,
        "Histogram") {
  reset_input_size(1);
  reset_output_size(1);
  low = low_;
  high = high_;
}

base::Status HistogramLayer::checkArgs() const {
  base::Status status;
  tensor::Tensor input = this->get_input(0);
  tensor::Tensor output = this->get_output(0);

  uint32_t input_ele_num = input.size();
  uint32_t output_ele_num = output.size();

  status = check_tensor_with_dim(input, device_type_,
                                 data_type_,
                                 input.get_dim(0));
  if (!status) {
    LOG(ERROR) << "The input tensor error in the histogram layer.";
    return status;
  }

  status = check_tensor_with_dim(output, device_type_,
                                 data_type_,
                                 output.get_dim(0));
  if (!status) {
    LOG(ERROR) << "The output tensor error in the histogram layer.";
    return status;
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  if (low > high) {
    LOG(ERROR) << "low > high in the histogram layer.";
    return status;
  }

  int32_t ele_num = output.get_dim(0);
  if ((high - low) > ele_num) {
    LOG(ERROR) << "(high_ - low) > ele_num in the histogram layer.";
    return status;
  }

  for (int i = 0; i < ele_num; ++i) {
    if (output.at<int32_t>(i) != 0) {
      LOG(ERROR) << "output.at<int32_t>(i) != 0 in the histogram layer.";
      return status;
    }
  }

  return base::error::Success();
}

base::Status HistogramLayer::compute() {
  auto input = this->get_input(0);
  auto output = this->get_output(0);

  para::histogram_para para;
  para.low = low;
  para.high = high;

  para.ele_num = input.get_dim(0);

  kernel::get_histogram_kernel(device_type_)(input, output, para,
                                             cuda_config_ ?
                                             cuda_config_->stream : nullptr);

  return base::error::Success();
}

base::Status HistogramLayer::forward() {
  if (apiTraceEnabled) {
    api_trace::API_trace trace("HistogramLayer::forward()");
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
