#include "swiglu_kernel.h"

namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1,
                       const tensor::Tensor& input2,
                       tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  CHECK(input1.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

  const float* input1_ptr = input1.ptr<float>();
  const float* input2_ptr = input2.ptr<float>();
  float* output_ptr = output.ptr<float>();

  std::vector<float> sigmoid_ptr;
  sigmoid_ptr.resize(input1.size());
  for (int i = 0; i < input1.size(); ++i) {
    sigmoid_ptr[i] = 1.0f / (1.0f + std::exp(-input1_ptr[i]));
  }

  for (int i = 0; i < input1.size(); ++i) {
    sigmoid_ptr[i] *= input1_ptr[i];
    output_ptr[i] = sigmoid_ptr[i] * input2_ptr[i];
  }
}
} // namespace kernel
