#include <cstdlib>
#include <glog/logging.h>
#include "base/alloc.h"

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() :
  DeviceAllocator(DeviceType::kDeviceCPU) {
    printf("enter here\n");
  }

void* CPUDeviceAllocator::allocate(size_t total_size) const {
  if (!total_size) {
    return nullptr;
  }

  void* data = malloc(total_size);
  return data;
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

std::shared_ptr<CPUDeviceAllocator>
  CPUDeviceAllocatorFactory::instance = nullptr;
}; // namespace base