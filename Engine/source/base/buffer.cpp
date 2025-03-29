#include <glog/logging.h>
#include "base/buffer.h"

namespace base {
Buffer::Buffer(size_t total_size,
  std::shared_ptr<DeviceAllocator> allocator, void* ptr, bool ues_external) :
  total_size_(total_size),
  ptr_(ptr),
  use_external_(ues_external),
  allocator_(allocator) {
  if (!ptr_ && allocator_) {
    use_external_ = false;
    ptr_ = allocator_->allocate(total_size);
    device_type_ = allocator_->device_type();
  }
}

Buffer::~Buffer() {
  if (!use_external_) {
    if (ptr_ && allocator_) {
      allocator_->release(ptr_);
      ptr_ = nullptr;
    }
  }
}

bool Buffer::allocate() {
  if (allocator_ && total_size_ != 0) {
    use_external_ = false;
    ptr_ = allocator_->allocate(total_size_);
    if (!ptr_) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

void Buffer::copy_from(const Buffer& buffer) const {
  CHECK(allocator_ != nullptr);
  CHECK(buffer.ptr_ != nullptr);

  size_t total_size = total_size_ < buffer.total_size_ ?
                      total_size_ :
                      buffer.total_size_;
  const DeviceType& buffer_device = buffer.device_type();
  const DeviceType& current_device = this->device_type();
  CHECK(buffer_device != DeviceType::kDeviceUnknown &&
        current_device != DeviceType::kDeviceUnknown);

  if (buffer_device == DeviceType::kDeviceCPU &&
      current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, total_size);
  } else if (buffer_device == DeviceType::kDeviceCUDA &&
             current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, total_size,
                              MemcpyKind::kMemcpyCUDA2CPU);
  } else if (buffer_device == DeviceType::kDeviceCPU &&
             current_device == DeviceType::kDeviceCUDA) {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, total_size,
                              MemcpyKind::kMemcpyCPU2CUDA);
  } else {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, total_size,
                              MemcpyKind::kMemcpyCUDA2CUDA);
  }
}

void Buffer::copy_from(const Buffer* buffer) const {
  CHECK(allocator_ != nullptr);
  CHECK(buffer != nullptr || buffer->ptr_ != nullptr);

  size_t dst_size = total_size_;
  size_t src_size = buffer->total_size_;
  size_t total_size = src_size < dst_size ? src_size : dst_size;

  const DeviceType& buffer_device = buffer->device_type();
  const DeviceType& current_device = this->device_type();
  CHECK(buffer_device != DeviceType::kDeviceUnknown &&
        current_device != DeviceType::kDeviceUnknown);

  if (buffer_device == DeviceType::kDeviceCPU &&
      current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, total_size);
  } else if (buffer_device == DeviceType::kDeviceCUDA &&
             current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, total_size,
                              MemcpyKind::kMemcpyCUDA2CPU);
  } else if (buffer_device == DeviceType::kDeviceCPU &&
             current_device == DeviceType::kDeviceCUDA) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, total_size,
                              MemcpyKind::kMemcpyCPU2CUDA);
  } else {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, total_size,
                              MemcpyKind::kMemcpyCUDA2CUDA);
  }
}

size_t Buffer::total_size() const {
  return total_size_;
}

void* Buffer::ptr() {
  return ptr_;
}

const void* Buffer::ptr() const {
  return ptr_;
}

bool Buffer::is_external() const {
  return this->use_external_;
}

DeviceType Buffer::device_type() const {
  return device_type_;
}

void Buffer::set_device_type(DeviceType device_type) {
  device_type_ = device_type;
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
  return allocator_;
}

std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
  return shared_from_this();
}
} // namespace base
