#ifndef SOFTMAX_KERNEL_H
#define SOFTMAX_KERNEL_H
#include <armadillo>
#include "tensor/tensor.h"
namespace kernel {
void softmax_inplace_cpu(const tensor::Tensor& input, void* stream = nullptr);
}  // namespace kernel
#endif // SOFTMAX_KERNEL_H
