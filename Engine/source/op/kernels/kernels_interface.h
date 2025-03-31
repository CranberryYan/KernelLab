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

typedef void (*AddKernel)(const tensor::Tensor& input1,
                          const tensor::Tensor& input2,
                          tensor::Tensor& output, void* stream);

AddKernel get_add_kernel(base::DeviceType device_type);

typedef void (*EmbeddingKernel)(const tensor::Tensor& input,
                                const tensor::Tensor& weight,
                                tensor::Tensor& output,
                                int32_t vocab_size, void* stream);
EmbeddingKernel get_embedding_kernel(base::DeviceType device_type);

typedef void (*MatmulKernel)(const tensor::Tensor& input,
                             const tensor::Tensor& weight,
                             tensor::Tensor& output,
                             float scale,
                             const CudaConfig* config);
MatmulKernel get_matmul_kernel(base::DeviceType device_type);

typedef void (*MatmulKernelQuant)(const tensor::Tensor& input,
                                  const tensor::Tensor& weight,
                                  tensor::Tensor& output,
                                  int32_t group_size,
                                  const tensor::Tensor& scale,
                                  const CudaConfig* config);
MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

typedef void (*SwigluKernel)(const tensor::Tensor& input1,
                             const tensor::Tensor& input2,
                             tensor::Tensor& output, void* stream);
SwigluKernel get_swiglu_kernel(base::DeviceType device_type,
                               void* stream = nullptr);
} // namespace kernel
#endif // KERNELS_INTERFACE_H
