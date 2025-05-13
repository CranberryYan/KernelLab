#include "embedding_kernel.h"

namespace kernel {
void embedding_kernel_cpu(const tensor::Tensor& input,
                          const tensor::Tensor& weight,
                          tensor::Tensor& output,
                          int32_t vocab_size, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(weight.device_type() == output.device_type());
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);

#if 0
eg:
  E = {[0.1, 0.2, 0.3],
       [0.4, 0.5, 0.6],
       [0.7, 0.8, 0.9],
       [1.0, 1.1, 1.2],
       [1.3, 1.4, 1.5],
       [1.6, 1.7, 1.8],
       [2.0, 2.1, 2.2]}
我们的词向量表, 5行3列
7行: 7个token(hi is my name yst . ?)
input: hi my name is yst.
input: 0, 2, 3, 1, 6
output: {[0.1, 0.2, 0.3],
         [0.7, 0.8, 0.9],
         [1.0, 1.1, 1.2],
         [0.4, 0.5, 0.6],
         [2.0, 2.1, 2.2]}
#endif

  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1); // 每个token的嵌入向量的维度
  const std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  for (int i = 0; i < input_num; ++i) {
    int token = *input.ptr<int32_t>(i); // 此时的input是token对应的单词表的序号
    if (token > vocab_size) {
      LOG(FATAL) << "Token index is greater than vocab size.";
    } else {
      float* dst = output.ptr<float>(i * weight_dim);
      const float* src = weight.ptr<const float>(token * weight_dim);
      if (weight.device_type() == base::DeviceType::kDeviceCPU) {
        alloc_cpu->memcpy(src, dst, weight_dim * sizeof(float),
                          base::MemcpyKind::kMemcpyCPU2CPU);
      } else {
        LOG(FATAL) << "Unknown device type of weight tensor in the embedding layer.";
      }
    }
  }
}
} // namespace kernel
