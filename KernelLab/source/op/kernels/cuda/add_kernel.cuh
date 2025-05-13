#ifndef ADD_KERNEL_CU_CUH
#define ADD_KERNEL_CU_CUH
#include "tensor/tensor.h"

namespace kernel {
void add_kernel_cu(const tensor::Tensor& input1,
                   const tensor::Tensor& input2,
                   tensor::Tensor& output,
                   para::add_para para,
                   void* stream = nullptr);
} // namespace kernel
#endif // ADD_KERNEL_CU_CUH
