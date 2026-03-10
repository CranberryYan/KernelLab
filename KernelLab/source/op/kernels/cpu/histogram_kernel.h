#ifndef HISTOGRAM_KERNEL_CPU_H
#define HISTOGRAM_KERNEL_CPU_H
#include "tensor/tensor.h"

namespace kernel {
void histogram_kernel_cpu(const tensor::Tensor& input,
                          tensor::Tensor& output,
                          para::histogram_para para,
                          void* stream);
} // namespace kernel
#endif // HISTOGRAM_KERNEL_CPU_H
