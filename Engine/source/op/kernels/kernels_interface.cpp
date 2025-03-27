#include <base/base.h>
#include "kernels_interface.h"
#include "cpu/rmsnorm_kernel.h"
#include "cuda/rmsnorm_kernel.cuh"

namespace kernel
{
RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rmsnorm_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rmsnorm_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
    return nullptr;
  }
}
} // namespace kernel
