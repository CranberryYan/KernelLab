#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../source/op/kernels/kernels_interface.h"

#if 1
TEST(test_cu_add, add_nostream) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 1;
  std::vector<int32_t> dims = {256, 4096};
  for (auto& dim : dims) {
    size *= dim;
  }

  tensor::Tensor input1_cpu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input2_cpu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input1_cu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cu, nullptr);
  tensor::Tensor input2_cu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    float input1_tmp = dist(mt);
    float input2_tmp = dist(mt);
    input1_cpu.set_value<float>(input1_tmp, i);
    input2_cpu.set_value<float>(input2_tmp, i);
    input1_cu.set_value<float>(input1_tmp, i);
    input2_cu.set_value<float>(input2_tmp, i);
  }

  kernel::get_add_kernel(base::DeviceType::kDeviceCPU)(
    input1_cpu, input2_cpu, output_cpu, nullptr);
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(
    input1_cu, input2_cu, output_cu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output_cpu.at<float>(i), output_cu.at<float>(i))
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
  }
}

TEST(test_cu_add, add_stream) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 1;
  std::vector<int32_t> dims = {256, 4096};
  for (auto& dim : dims) {
    size *= dim;
  }

  tensor::Tensor input1_cpu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input2_cpu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input1_cu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cu, nullptr);
  tensor::Tensor input2_cu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, size,
                         true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    float input1_tmp = dist(mt);
    float input2_tmp = dist(mt);
    input1_cpu.set_value<float>(input1_tmp, i);
    input2_cpu.set_value<float>(input2_tmp, i);
    input1_cu.set_value<float>(input1_tmp, i);
    input2_cu.set_value<float>(input2_tmp, i);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  kernel::get_add_kernel(base::DeviceType::kDeviceCPU)(
    input1_cpu, input2_cpu, output_cpu, nullptr);
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(
    input1_cu, input2_cu, output_cu, stream);

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output_cpu.at<float>(i), output_cu.at<float>(i))
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
  }
}
#endif
