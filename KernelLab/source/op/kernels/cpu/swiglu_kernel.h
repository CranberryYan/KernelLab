#ifndef SWIGLU_KERNEL_CPU_H
#define SWIGLU_KERNEL_CPU_H
#include "tensor/tensor.h"

namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1,
                       const tensor::Tensor& input2,
                       tensor::Tensor& output, void* stream);
} // namespace kernel
#endif // SWIGLU_KERNEL_CPU_H
