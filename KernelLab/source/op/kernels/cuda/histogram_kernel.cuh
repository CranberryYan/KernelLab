#ifndef HISTOGRAM_KERNEL_H_
#define HISTOGRAM_KERNEL_H_
#include "tensor/tensor.h"
#include "../kernels_interface.h"

namespace kernel {
void histogram_kernel_cu(const tensor::Tensor &input,
                         tensor::Tensor &output,
                         para::histogram_para para,
                         void* stream);
} // namespace kernel
#endif // HISTOGRAM_KERNEL_H_
