#include <cstdlib>
#include "op/scatter.h"
#include "./base/API_trace.h"
#include "kernels/kernels_interface.h"

namespace op {
static const bool apiTraceEnabled = (std::getenv("api_trace") != nullptr);

ScatterLayer::ScatterLayer(base::DeviceType device_type) :
  Layer(device_type, LayerType::kLayerScatter, "Scatter") {
  reset_input_size(3); // input src index
  reset_output_size(1);
}

base::Status ScatterLayer::checkArgs() const {
  base::Status status;
  tensor::Tensor input = this->get_input(0);
  tensor::Tensor src = this->get_input(1);
  tensor::Tensor index = this->get_input(2);
  tensor::Tensor output = this->get_output(0);

  uint32_t input_ele_num = input.size();
  uint32_t index_ele_num = index.size();
  uint32_t output_ele_num = output.size();

  if (op_type == para::ScatterOpType::Scatter_Add ||
      op_type == para::ScatterOpType::Scatter_Update) {
    status = check_tensor_with_dim(src, device_type_,
                                  data_type_,
                                  src.get_dim(0),
                                  src.get_dim(1));;
    if (!status) {
      LOG(ERROR) << "The src tensor error in the scatter layer.";
      return status;
    }

    if (input.get_buffer() != output.get_buffer()) {
      return base::error::InvalidArgument("input.buffer != output.buffer");
    }

    if (input_ele_num != output_ele_num) {
      return base::error::InvalidArgument(
        "The input tensor has a wrong ele num.");
    }

    status = check_tensor_with_dim(index, device_type_,
                                  base::DataType::kDataTypeInt32,
                                  index.get_dim(0),
                                  index.get_dim(1));
    if (!status) {
      LOG(ERROR) << "The index tensor error in the scatter layer.";
      return status;
    }

    status = check_tensor_with_dim(output, device_type_,
                                  data_type_,
                                  output.get_dim(0),
                                  output.get_dim(1));
    if (!status) {
      LOG(ERROR) << "The output tensor error in the scatter layer.";
      return status;
    }
  } else if (op_type == para::ScatterOpType::Gather) {
    if (input.get_buffer() == output.get_buffer()) {
      return base::error::InvalidArgument("input.buffer == output.buffer");
    }

    if (index_ele_num != output_ele_num) {
      return base::error::InvalidArgument(
        "The output tensor has a wrong ele num.");
    }
  }

  status = check_tensor_with_dim(input, device_type_,
                                 data_type_,
                                 input.get_dim(0),
                                 input.get_dim(1));
  if (!status) {
    LOG(ERROR) << "The input tensor error in the scatter layer.";
    return status;
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  return base::error::Success();
}

base::Status ScatterLayer::compute() {
  auto input = this->get_input(0);
  auto src = this->get_input(1);
  auto index = this->get_input(2);
  auto output = this->get_output(0);

  para::scatter_para para;
  para.op_type = this->op_type;
  para.block_num = index.get_dim(0); // TODO: 小shape下性能不佳 -> 超发
  para.thread_num = index.get_dim(1);
  para.input_dims = input.dims();
  para.index_dims = index.dims();
  para.src_dims = src.dims();
  para.input_rows = para.input_dims[0];
  para.input_cols = para.input_dims[1];
  uint32_t index_ele_num = 1;
  for (auto &dim : para.index_dims) {
    index_ele_num *= dim;
  }
  para.index_ele_num = index_ele_num;
  para.index_ele_num_per_block = index.get_dim(1);

  uint32_t input_ele_num = 1;
  for (auto &dim : para.input_dims) {
    input_ele_num *= dim;
  }
  para.input_ele_num = input_ele_num;
  para.input_ele_num_per_block = input.get_dim(1);

  uint32_t src_ele_num = 1;
  for (auto &dim : para.src_dims) {
    src_ele_num *= dim;
  }
  para.src_ele_num = src_ele_num;
  para.src_ele_num_per_block = src.get_dim(1);

  kernel::get_scatter_kernel(device_type_)(input, src, index, output, para,
                                           cuda_config_ ?
                                           cuda_config_->stream : nullptr);

  return base::error::Success();
}

base::Status ScatterLayer::forward() {
  if (apiTraceEnabled) {
    api_trace::API_trace trace("ScatterLayer::forward()");
    trace.set_tensor("input", this->inputs_[0]);
    trace.set_tensor("src", this->inputs_[1]);
    trace.set_tensor("index", this->inputs_[2]);
    trace.set_tensor("output", this->outputs_[0]);
    trace.print_scatter_type(op_type);

    trace.print_tensor();
  }

  base::Status status = this->checkArgs();
  if (!status) {
    // return status;
  }

  status = this->compute();
  if (!status) {
    return status;
  }

  return base::error::Success();
}
} // namespace op
