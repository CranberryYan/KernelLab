#ifndef SOFTMAX_KERNEL_CU_H_
#define SOFTMAX_KERNEL_CU_H_
#include "tensor/tensor.h"

namespace kernel {
void softmax_kernel_cu(const tensor::Tensor& input,
                        tensor::Tensor& output,
                        para::softmax_para para,
                        void* stream = nullptr);
} // namespace kernel
#endif // SOFTMAX_KERNEL_CU_H_
