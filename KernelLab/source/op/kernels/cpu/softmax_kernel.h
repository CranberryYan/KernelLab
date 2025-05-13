#ifndef SOFTMAX_KERNEL_H
#define SOFTMAX_KERNEL_H
#include <armadillo>
#include "tensor/tensor.h"

namespace kernel {
void softmax_kernel_cpu(const tensor::Tensor& input,
                        tensor::Tensor& output,
                        para::softmax_para para,
                        void* stream = nullptr);
}  // namespace kernel
#endif // SOFTMAX_KERNEL_H
