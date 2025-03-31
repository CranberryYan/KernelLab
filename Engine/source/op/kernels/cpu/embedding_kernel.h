#ifndef EMBEDDING_KERNEL_CPU_H
#define EMBEDDING_KERNEL_CPU_H
#include "base/base.h"
#include "tensor/tensor.h"
namespace kernel {
void embedding_kernel_cpu(const tensor::Tensor& input,
                          const tensor::Tensor& weight,
                          tensor::Tensor& output,
                          int32_t vocab_size, void* stream = nullptr);
}  // namespace kernel
#endif  // EMBEDDING_KERNEL_CPU_H
