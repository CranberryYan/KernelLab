#ifndef KERNELLAB_INCLUDE_TENSOR_H_
#define KERNELLAB_INCLUDE_TENSOR_H_
#include <cmath>
#include <memory>
#include <vector>
#include <random>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "base/para.h"
#include "base/base.h"
#include "base/buffer.h"

namespace tensor {
class Tensor {
public:
  explicit Tensor() = default;
  explicit Tensor(base::DataType data_type, int32_t dim0,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);
  explicit Tensor(base::DataType data_type,
                  int32_t dim0, int32_t dim1, int32_t dim2,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);
  explicit Tensor(base::DataType data_type,
                  int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);
  explicit Tensor(base::DataType data_type,
                  std::vector<int32_t> dims, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);

  void to_cpu();
  void to_cuda(cudaStream_t stream = nullptr);
  bool is_empty() const;
  void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                   base::DataType data_type,
                   bool need_alloc, void* ptr);

  size_t size() const;
  size_t total_size() const;
  int32_t dims_size() const;

  template <typename T> T* ptr();
  template <typename T> const T* ptr() const;
  template <typename T> T* ptr(int64_t index);
  template <typename T> const T* ptr(int64_t index) const;

  template <typename T> T& at(int64_t offset);
  template <typename T> const T& at(int64_t offset) const;

  template <typename T> T& index(int64_t offset);
  template <typename T> const T& index(int64_t offset) const;

  template <typename T> bool set_value(T value, int64_t offset);

  int32_t get_dim(int32_t idx) const;
  const std::vector<int32_t>& dims() const;

  base::DeviceType device_type() const;
  void set_device_type(base::DeviceType device_type) const;

  base::DataType data_type() const;

  std::vector<size_t> strides() const;

  std::shared_ptr<base::Buffer> get_buffer() const;

  bool assign(std::shared_ptr<base::Buffer> buffer);

  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                bool need_realloc = false);

  tensor::Tensor clone() const;

  void reshape(const std::vector<int32_t>& dims);
private:
  size_t size_ = 0;
  std::vector<int32_t> dims_;
  std::shared_ptr<base::Buffer> buffer_;
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;
};

template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
T& Tensor::at(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());

  // device检查
  T* ptr_h = nullptr;
  if (this->device_type() == base::DeviceType::kDeviceCPU) {
    T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
  } else if (this->device_type() == base::DeviceType::kDeviceCUDA) {
    ptr_h = reinterpret_cast<T*>(malloc(sizeof(T)));
    cudaMemcpy(ptr_h,
               reinterpret_cast<T*>(buffer_->ptr()) + offset,
               sizeof(T),
               cudaMemcpyDeviceToHost);
    return *ptr_h;
  }

  // Return a default value or handle the error for unrecognized device types
  LOG(FATAL) << "Unrecognized device type.";
  static T default_value = -1.0f;
  return default_value;
}

template <typename T>
const T& Tensor::at(int64_t offset) const {
  // 合法性检查
  // 0 < offset < size
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());

  // device检查
  if (this->device_type() == base::DeviceType::kDeviceCPU) {
    T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
  } else if (this->device_type() == base::DeviceType::kDeviceCUDA) {
    T* ptr_h = reinterpret_cast<T*>(malloc(sizeof(T)));
    cudaMemcpy(ptr_h,
               reinterpret_cast<T*>(buffer_->ptr()) + offset,
               sizeof(T),
               cudaMemcpyDeviceToHost);
    return *ptr_h;
  }

  // Return a default value or handle the error for unrecognized device types
  LOG(FATAL) << "Unrecognized device type.";
  static T default_value = -1.0f;
  return default_value;
}

template <typename T>
bool Tensor::set_value(T value, int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());

  if (this->device_type() == base::DeviceType::kDeviceCPU) {
    *(reinterpret_cast<T*>(buffer_->ptr()) + offset) = value;
    return true;
  } else if (this->device_type() == base::DeviceType::kDeviceCUDA) {
    cudaMemcpy(reinterpret_cast<T*>(buffer_->ptr()) + offset,
               &value,
               sizeof(T),
               cudaMemcpyHostToDevice);
    return true;
  }

  return false;
}

} // namespace tensor
#endif // KERNELLAB_INCLUDE_TENSOR_H_
