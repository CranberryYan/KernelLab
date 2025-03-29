#ifndef ENGINE_INCLUDE_BASE_BUFFER_H_
#define ENGINE_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"
namespace base {
// use_external_:
//  ptr_的来源有两种, 外部直接赋值, Buffer不需要进行管理, use_external_为true
//  Buffer需要进行管理, use_external_为false, 没人使用该buffer的时候会自动将
//  ptr_指向的地址用对应类型的Allocator完成释放
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
public:
  explicit Buffer() = default;
  explicit Buffer(size_t total_size,
                  std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr,
                  bool ues_external = false);
  virtual ~Buffer();

  bool allocate();
  void copy_from(const Buffer& buffer) const;
  void copy_from(const Buffer* buffer) const;

  size_t total_size() const;
  void* ptr();
  const void* ptr() const;
  bool is_external() const;
  DeviceType device_type() const;
  void set_device_type(DeviceType device_type);
  std::shared_ptr<DeviceAllocator> allocator() const;
  std::shared_ptr<Buffer> get_shared_from_this();
private:
  size_t total_size_ = 0;
  void* ptr_ = nullptr;
  bool use_external_ = false;
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
  std::shared_ptr<DeviceAllocator> allocator_;
};
}  // namespace base
#endif
