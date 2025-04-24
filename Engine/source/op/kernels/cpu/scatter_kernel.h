#ifndef SCATTER_KERNEL_CPU_H_
#define SCATTER_KERNEL_CPU_H_
#include "tensor/tensor.h"

namespace kernel {
void scatter_kernel_cpu(const tensor::Tensor &input,
                        const tensor::Tensor &src,
                        const tensor::Tensor &index,
                        tensor::Tensor &output,
                        para::scatter_para para,
                        void* stream = nullptr);
}
#endif // SCATTER_KERNEL_CPU_H_