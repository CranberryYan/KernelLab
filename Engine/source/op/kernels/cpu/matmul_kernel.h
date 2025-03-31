#ifndef MATMUL_KERNEL_CPU_CH
#define MATMUL_KERNEL_CPU_CH
#include "base/cuda_config.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input,
                       const tensor::Tensor& weight,
                       tensor::Tensor& output,
                       float scale = 1.f,
                       const CudaConfig* config = nullptr);
}  // namespace kernel
#endif  // MATMUL_KERNEL_CPU_H
