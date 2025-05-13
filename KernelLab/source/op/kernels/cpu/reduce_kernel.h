#ifndef REDUCE_KERNEL_CPU_H_
#define REDUCE_KERNEL_CPU_H_
#include "base/base.h"
#include "tensor/tensor.h"

namespace kernel {
void reduce_kernel_cpu(const tensor::Tensor &input,
                       tensor::Tensor &output,
                       para::reduce_para para,
                       void* stream = nullptr);
} // namespace kernel
#endif // REDUCE_KERNEL_CPU_H_