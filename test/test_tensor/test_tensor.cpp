#include <vector>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "tensor/tensor.h"

TEST(test_tensor, init0) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  int32_t dim = 256;
  tensor::Tensor t1_cu(base::DataType::kDataTypeFp32,
                       dim, true, alloc_cu, nullptr);
  tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32,
                        dim, true, alloc_cpu, nullptr);

  ASSERT_EQ(t1_cu.is_empty(), false);
  ASSERT_EQ(t1_cpu.is_empty(), false);
}

TEST(test_tensor, init1) {
  float* ptr = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 512));
  for (int i = 0; i < 256 * 512; ++i) {
    ptr[i] = i;
  }
  tensor::Tensor t1(base::DataType::kDataTypeFp32,
                    256, 512, false, nullptr, ptr);
  ASSERT_EQ(t1.device_type(), base::DeviceType::kDeviceUnknown);
}

TEST(test_tensor, init2) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  std::vector<int32_t> dims = {256, 4096};

  tensor::Tensor t1(base::DataType::kDataTypeFp32,
    dims, true, alloc_cu, nullptr);
  set_value_cu(t1.ptr<float>(), 256 * 4096, 1.0f);
  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t1.device_type(), base::DeviceType::kDeviceCUDA);
  ASSERT_EQ(t1.at<float>(512), 1.0f);

  t1.to_cpu();
  ASSERT_EQ(t1.device_type(), base::DeviceType::kDeviceCPU);
  ASSERT_EQ(t1.at<float>(512), 1.0f);
}

TEST(test_tensor, clone_cpu) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  for (int i = 0; i < 32 * 32; ++i) {
    t1_cpu.at<float>(i) = 1.f;
  }

  tensor::Tensor t2_cpu = t1_cpu.clone();
  float* p2 = new float[32 * 32];
  std::memcpy(p2, t2_cpu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  std::memcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }
  delete[] p2;
}

TEST(test_tensor, clone_cuda) {
  using namespace base;
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
  ASSERT_EQ(t1_cu.is_empty(), false);
  set_value_cu(t1_cu.ptr<float>(), 32 * 32, 1.f);

  tensor::Tensor t2_cu = t1_cu.clone();
  float* p2 = new float[32 * 32];
  cudaMemcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  cudaMemcpy(p2, t1_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  ASSERT_EQ(t2_cu.data_type(), base::DataType::kDataTypeFp32);
  ASSERT_EQ(t2_cu.size(), 32 * 32);

  t2_cu.to_cpu();
  std::memcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }
  delete[] p2;
}

TEST(test_tensor, to_cpu) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu(
    base::DataType::kDataTypeFp32, 32, 32, true, alloc_cu, nullptr);
  ASSERT_EQ(t1_cu.is_empty(), false);
  ASSERT_EQ(t1_cu.device_type(), base::DeviceType::kDeviceCUDA);

  set_value_cu(t1_cu.ptr<float>(), 32 * 32, 1.0f);

  t1_cu.to_cpu();
  ASSERT_EQ(t1_cu.device_type(), base::DeviceType::kDeviceCPU);

  float* cpu_ptr = t1_cu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(cpu_ptr + i), 1.f);
  }
}

TEST(test_tensor, to_cuda) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(
    base::DataType::kDataTypeFp32, 32, 32, true, alloc_cpu, nullptr);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  ASSERT_EQ(t1_cpu.device_type(), base::DeviceType::kDeviceCPU);

  float* p1 = t1_cpu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    p1[i] = 1.0f;
  }

  t1_cpu.to_cuda();
  float* p2 = reinterpret_cast<float*>(malloc(32 * 32 * sizeof(float)));
  cudaMemcpy(
    p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.0f);
  }

  free(p2);
}

TEST(test_tensor, assign1) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);

  int32_t size = 32 * 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }
  std::shared_ptr<Buffer> buffer =
      std::make_shared<Buffer>(size * sizeof(float), nullptr, ptr, true);
  buffer->set_device_type(DeviceType::kDeviceCPU);

  ASSERT_EQ(t1_cpu.assign(buffer), true);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  ASSERT_NE(t1_cpu.ptr<float>(), nullptr);
  delete[] ptr;
}
