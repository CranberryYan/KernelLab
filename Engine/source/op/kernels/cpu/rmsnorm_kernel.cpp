#include "rmsnorm_kernel.h"

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input,
                        const tensor::Tensor& weight,
                        tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);

  const float* input_ptr = input.ptr<float>();
  float* output_ptr = output.ptr<float>();
  const int32_t total_num = static_cast<int32_t>(input.size());

// RMSNorm 公式
//  scale = Sigma(x^2) / d
//  rsqrt = 1 / (scale + eps) ^ 0.5
//  y = x * rsqrt * w
  float sum = 0.0f;
  for (int i = 0; i < total_num; ++i) {
    float input_value = input.index<float>(i);
    sum += std::pow(input_value, 2);
  }

#ifdef QWEN2_SUPPORT
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif

  float scale = sum / total_num;

  float rsqrt = 1.0f / std::sqrt(scale + eps);
  for (int i = 0; i < total_num; ++i) {
    output_ptr[i] = input_ptr[i] * rsqrt * weight.index<float>(i);
  }
}
} // namespace kernel