#ifndef EMBEDDING_KERNEL_CU_CUH
#define EMBEDDING_KERNEL_CU_CUH
#include "tensor/tensor.h"
namespace kernel {
void embedding_kernel_cu(const tensor::Tensor& input,
                         const tensor::Tensor& weight,
                         tensor::Tensor& output,
                         int32_t vocab_size, void* stream = nullptr);
}
#endif // EMBEDDING_KERNEL_CU_CUH
