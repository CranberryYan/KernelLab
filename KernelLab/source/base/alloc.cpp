#include <cuda_runtime_api.h>
#include "base/alloc.h"

namespace base {
void DeviceAllocator::memcpy(const void* src_ptr, void* dst_ptr,
  size_t total_size, MemcpyKind memcpy_kind, void* stream, bool need_sync) const
{
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dst_ptr, nullptr);
  if (!total_size) {
    printf("Error: total_size is 0 in DeviceAllocator::memcpy\n");
    return;
  }

  cudaStream_t stream_ = nullptr;
  if (stream != nullptr) {
    stream_ = static_cast<cudaStream_t>(stream);
  }

  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(dst_ptr, src_ptr, total_size);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      cudaMemcpy(dst_ptr, src_ptr, total_size, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(dst_ptr, src_ptr, total_size, cudaMemcpyHostToDevice, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    if (!stream_) {
      cudaMemcpy(dst_ptr, src_ptr, total_size, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(dst_ptr, src_ptr, total_size, cudaMemcpyDeviceToHost, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    if (!stream_) {
      cudaMemcpy(dst_ptr, src_ptr, total_size, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(dst_ptr, src_ptr, total_size, cudaMemcpyDeviceToDevice, stream_);
    }
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
  }
  if (need_sync) {
    cudaDeviceSynchronize();
  }
}

void DeviceAllocator::memset_zero(void* ptr, size_t total_size,
  void* stream, bool need_sync) {
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, total_size);
  } else {
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      cudaMemsetAsync(ptr, 0, total_size, stream_);
    } else {
      cudaMemset(ptr, 0, total_size);
    }
    if (need_sync) {
      cudaDeviceSynchronize();
    }
  }
}
} // namespace base
