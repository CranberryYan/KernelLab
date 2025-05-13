#ifndef REDUCE_KERNEL_H_
#define REDUCE_KERNEL_H_
#include "tensor/tensor.h"
#include "../kernels_interface.h"

namespace kernel {
void reduce_kernel_cu(const tensor::Tensor &input,
                      tensor::Tensor &output,
                      para::reduce_para para,
                      void* stream);
} // namespace kernel
#endif // REDUCE_KERNEL_H_
