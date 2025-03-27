#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H

#include <base/cuda_config.h>
#include "tensor/tensor.h"

namespace kernel
{
typedef void (*RMSNormKernel)(const tensor::Tensor& input,
                              const tensor::Tensor& weight,
                              tensor::Tensor& output, void* stream);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);
} // namespace kernel
#endif // KERNELS_INTERFACE_H
