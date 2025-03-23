#ifndef ENGINE_INCLUDE_TENSOR_H_
#define ENGINE_INCLUDE_TENSOR_H_
#include <memory>
#include <vector>
#include <armadillo>
#include <driver_types.h>
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

  template <typename T> T& index(int64_t offset);
  template <typename T> const T& index(int64_t offset) const;

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
               reinterpret_cast<T*>(this->ptr<T>()) + offset,
               sizeof(T),
               cudaMemcpyDeviceToHost);
    return *ptr_h;
  }
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
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
               reinterpret_cast<T*>(this->ptr<T>()) + offset,
               sizeof(T),
               cudaMemcpyDeviceToHost);
    return *ptr_h;
  }
}
} // namespace tensor
#endif // ENGINE_INCLUDE_TENSOR_H_
