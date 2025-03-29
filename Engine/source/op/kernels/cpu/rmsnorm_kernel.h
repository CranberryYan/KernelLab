#ifndef RMSNORM_KERNEL_CPU_CUH
#define RMSNORM_KERNEL_CPU_CUH
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input,
                        const tensor::Tensor& weight,
                        tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif  // RMSNORM_KERNEL_CPU_CUH
