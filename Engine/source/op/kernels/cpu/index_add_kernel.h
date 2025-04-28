#ifndef INDEX_ADD_KERNEL_CPU_H
#define INDEX_ADD_KERNEL_CPU_H
#include "tensor/tensor.h"

namespace kernel {
void index_add_kernel_cpu(const tensor::Tensor& target,
                          const tensor::Tensor& index,
                          const tensor::Tensor& source,
                          tensor::Tensor& output,
                          para::index_add_para para,
                          void* stream);
} // namespace kernel
#endif // INDEX_ADD_KERNEL_CPU_H