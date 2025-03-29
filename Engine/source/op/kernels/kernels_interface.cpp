#include <base/base.h>
#include "kernels_interface.h"
#include "cpu/rmsnorm_kernel.h"
#include "cuda/rmsnorm_kernel.cuh"
#include "cpu/add_kernel.h"
#include "cuda/add_kernel.cuh"
#include "cpu/embedding_kernel.h"
#include "cuda/embedding_kernel.cuh"

namespace kernel
{
RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rmsnorm_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rmsnorm_kernel_cu;
  }
  LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
  return nullptr;
}

AddKernel get_add_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return add_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return add_kernel_cu;
  }
  LOG(FATAL) << "Unknown device type for get a add kernel.";
  return nullptr;
}

EmbeddingKernel get_embedding_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return embedding_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return embedding_kernel_cu;
  }
    LOG(FATAL) << "Unknown device type for get an embedding kernel.";
    return nullptr;
}
} // namespace kernel
