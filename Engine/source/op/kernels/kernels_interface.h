#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H

#include <base/cuda_config.h>
#include "tensor/tensor.h"

namespace kernel {
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

typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q,
                           const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos,
                           const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream);
RoPEKernel get_rope_kernel(base::DeviceType device_type);

typedef void (*MHAKernel)(int32_t pos, int32_t head_num, int32_t layer_index, 
                          int32_t seq_len,
                          int32_t kv_dim, int32_t kv_mul,
                          int32_t head_size,
                          tensor::Tensor& mha_out,
                          const tensor::Tensor& query_tensor,
                          const tensor::Tensor& score_tensor,
                          const tensor::Tensor& key_cache_tensor,
                          const tensor::Tensor& value_cache_tensor,
                          base::DeviceType device_type,
                          CudaConfig*);
MHAKernel get_mha_kernel(base::DeviceType device_type);

typedef void (*SoftmaxInplaceKernel)(const tensor::Tensor& input, void* stream);
SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type);

typedef void (*ScaleSumKernel)(const tensor::Tensor& value,
                               const tensor::Tensor& scale,
                               tensor::Tensor& output,
                               int t, int size, int stride,
                               void* stream);
ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);

typedef void (*ScaleKernel)(float scale,
                            const tensor::Tensor& input, void* stream);
ScaleKernel get_scale_kernel(base::DeviceType device_type);
} // namespace kernel
#endif // KERNELS_INTERFACE_H
