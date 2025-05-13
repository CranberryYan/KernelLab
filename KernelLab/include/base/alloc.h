#ifndef KERNELLAB_INCLUDE_BASE_ALLOC_H_
#define KERNELLAB_INCLUDE_BASE_ALLOC_H_
#include <map>
#include <memory>
#include "base.h"

namespace base {
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
public:
  explicit DeviceAllocator(DeviceType device_type) :
    device_type_(device_type) {}

  virtual DeviceType device_type() const {return device_type_;}

  virtual void release(void* ptr) const = 0;

  virtual void* allocate(size_t total_size) const = 0;

  virtual void memcpy(const void* src_ptr, void* dst_ptr,
                      size_t total_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
                      void* stream = nullptr, bool need_sync = false) const;

  virtual void memset_zero(void* ptr, size_t total_size,
                           void* stream, bool need_sync = false);

private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
public:
  explicit CPUDeviceAllocator();

  virtual void release(void* ptr) const override;

  virtual void* allocate(size_t total_size) const override;
};

class CPUDeviceAllocatorFactory {
public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

struct CudaMemoryBuffer {
public:
  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void* data, size_t total_size, bool busy) :
    data(data), total_size(total_size), busy(busy) {}

  void* data;
  size_t total_size;
  bool busy;
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
  explicit CUDADeviceAllocator() :
    DeviceAllocator(DeviceType::kDeviceCUDA) {}

  virtual void release(void* ptr) const override;

  virtual void* allocate(size_t total_size) const override;
private:
  mutable std::map<int, size_t> no_busy_cnt_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

class CUDADeviceAllocatorFactory {
public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }
private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};
} // namespace base
#endif // KERNELLAB_INCLUDE_BASE_ALLOC_H_
