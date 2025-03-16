#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"
#include "../utils.cuh"

TEST(test_buffer, allocate_cpu) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc =
    base::CPUDeviceAllocatorFactory::get_instance();
  base::Buffer buffer(32, alloc);
  ASSERT_NE(buffer.ptr(), nullptr);
}

TEST(test_buffer, allocate_cuda) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc =
    base::CUDADeviceAllocatorFactory::get_instance();
  base::Buffer buffer(32, alloc);
  ASSERT_NE(buffer.ptr(), nullptr);
}

TEST(test_buffer, use_external) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc =
    base::CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  base::Buffer buffer(32, nullptr, ptr, true);
  ASSERT_EQ(buffer.is_external(), true);
  delete[] ptr;
}

TEST(test_buffer, memcpy_CPU2CUDA) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_d =
    base::CUDADeviceAllocatorFactory::get_instance();

  int32_t total_num = 32;
  float* ptr = static_cast<float*>(malloc(total_num * sizeof(float)));
  float* ptr_d = static_cast<float*>(malloc(total_num * sizeof(float)));
  for(int i = 0; i < total_num; ++i) {
    ptr[i] = static_cast<float>(i);
  }

  // 在外部我们自行malloc的内存, 所以没有alloc, 并且直接传入已经malloc的ptr
  base::Buffer buffer_h(total_num * sizeof(float), nullptr, ptr, true);
  buffer_h.set_device_type(base::DeviceType::kDeviceCPU);
  ASSERT_EQ(buffer_h.is_external(), true);

  // 在外部并没有cudaMalloc, 所示会在内部malloc, 需要传入alloc, flase
  base::Buffer buffer_d(total_num * sizeof(float), alloc_d, nullptr, false);
  buffer_d.set_device_type(base::DeviceType::kDeviceCUDA);

  buffer_d.copy_from(buffer_h);
  cudaMemcpy(ptr_d, buffer_d.ptr(), total_num * sizeof(float),
    cudaMemcpyDeviceToHost);
  for (int i = 0; i < total_num; ++i) {
    ASSERT_EQ(ptr_d[i], float(i));
  }

  free(ptr);
  free(ptr_d);
}

TEST(test_buffer, memcpy_CUDA2CUDA) {
  auto alloc_d = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t total_num = 32;
  base::Buffer buffer_d_1(total_num * sizeof(float), alloc_d, nullptr, false);
  base::Buffer buffer_d_2(total_num * sizeof(float), alloc_d, nullptr, false);
  set_value_cu((float*)buffer_d_2.ptr(), total_num, 1.0);

  buffer_d_1.copy_from(buffer_d_2);
  float* ptr = static_cast<float*>(malloc(total_num * sizeof(float)));
  cudaMemcpy(ptr, buffer_d_1.ptr(), sizeof(float)  * total_num,
    cudaMemcpyDeviceToHost);
  for (int i = 0; i < total_num; ++i) {
    ASSERT_EQ(ptr[i], 1.f);
  }

  free(ptr);
}

TEST(test_buffer, memcpy_CUDA2CPU) {
  auto alloc_h = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_d = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t total_num = 32;
  base::Buffer buffer_h(total_num * sizeof(float), alloc_h, nullptr, false);
  ASSERT_EQ(buffer_h.device_type(), base::DeviceType::kDeviceCPU);
  base::Buffer buffer_d(total_num * sizeof(float), alloc_d, nullptr, false);
  ASSERT_EQ(buffer_d.device_type(), base::DeviceType::kDeviceCUDA);

  set_value_cu((float*)buffer_d.ptr(), total_num, 1.0);
  buffer_h.copy_from(buffer_d);

  float* ptr = static_cast<float*>(buffer_h.ptr());
  for (int i = 0; i < total_num; ++i) {
    ASSERT_EQ(ptr[i], 1.f);
  }
}