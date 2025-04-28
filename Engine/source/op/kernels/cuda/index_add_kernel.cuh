#ifndef INDEX_ADD_KERNEL_CU_H_
#define INDEX_ADD_KERNEL_CU_H_
#include "tensor/tensor.h"

namespace kernel {
void index_add_kernel_cu(const tensor::Tensor& target,
                         const tensor::Tensor& index,
                         const tensor::Tensor& source,
                         tensor::Tensor& output,
                         para::index_add_para para,
                         void* stream = nullptr);
} // namespace kernel
#endif // INDEX_ADD_KERNEL_CU_H_