#ifndef ADD_KERNEL_CPU_H
#define ADD_KERNEL_CPU_H
#include "tensor/tensor.h"
namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1,
                    const tensor::Tensor& input2,
                    tensor::Tensor& output,
                    para::add_para para,
                    void* stream = nullptr);
} // namespace kernel
#endif // ADD_KERNEL_CPU_H
