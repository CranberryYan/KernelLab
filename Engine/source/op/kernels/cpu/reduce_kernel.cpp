#include "reduce_kernel.h"

namespace kernel {
void reduce_kernel_cpu(const tensor::Tensor &input,
                       tensor::Tensor &output,
                       para::reduce_para para,
                       void* stream) {
  CHECK(!input.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(input.device_type() == output.device_type());

  uint32_t ele_num = para.ele_num;
  uint32_t after_reduce_num = para.after_reduce_num;
  uint32_t reduce_num_per_block =
    (ele_num + after_reduce_num - 1) / after_reduce_num;

  for (uint32_t i = 0; i < after_reduce_num; ++i) {
    output.at<float>(i) = 0.0f;
  }

  for (uint32_t i = 0; i < ele_num; ++i) {
    uint32_t block_id = i / reduce_num_per_block;
    output.at<float>(block_id) += input.at<float>(i);
  }
}
} // namespace kernel
