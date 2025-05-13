#ifndef SCATTER_KERNEL_CU_H_
#define SCATTER_KERNEL_CU_H_
#include "tensor/tensor.h"

namespace kernel {
void scatter_kernel_cu(const tensor::Tensor &input,
                       const tensor::Tensor &src,
                       const tensor::Tensor &index,
                       tensor::Tensor &output,
                       para::scatter_para para,
                       void* stream = nullptr);
} // namespace kernel
#endif // SCATTER_KERNEL_CU_H_
