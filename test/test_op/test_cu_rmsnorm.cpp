#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../source/op/kernels/kernels_interface.h"

#if 0
TEST(test_rmsnorm_cu, rmsnorm_nostream) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();

  int32_t total_num = 256 * 4096;

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cpu, nullptr);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cu, nullptr);
  tensor::Tensor weight_cu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < total_num; ++i) {
    float intput_tmp = dist(mt);
    float weight_tmp = dist(mt);
    input_cpu.set_value<float>(intput_tmp, i);
    weight_cpu.set_value<float>(weight_tmp, i);
    input_cu.set_value<float>(intput_tmp, i);
    weight_cu.set_value<float>(weight_tmp, i);
  }

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)
    (input_cpu, weight_cpu, output_cpu, nullptr);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)
    (input_cu, weight_cu, output_cu, nullptr);

  for (int i = 0; i < total_num; ++i) {
    ASSERT_NEAR(output_cu.index<float>(i), output_cpu.index<float>(i), 1e-3f)
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.index<float>(i), output_cu.index<float>(i));
  }
}

TEST(test_rmsnorm_cu, rmsnorm_stream) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();

  int32_t total_num = 256 * 4096;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cpu, nullptr);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cu, nullptr);
  tensor::Tensor weight_cu(base::DataType::kDataTypeFp32, total_num,
                           true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < total_num; ++i) {
    float intput_tmp = dist(mt);
    float weight_tmp = dist(mt);
    input_cpu.set_value<float>(intput_tmp, i);
    weight_cpu.set_value<float>(weight_tmp, i);
    input_cu.set_value<float>(intput_tmp, i);
    weight_cu.set_value<float>(weight_tmp, i);
  }

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)
    (input_cpu, weight_cpu, output_cpu, nullptr);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)
    (input_cu, weight_cu, output_cu, stream);

  for (int i = 0; i < total_num; ++i) {
    ASSERT_NEAR(output_cu.index<float>(i), output_cpu.index<float>(i), 1e-3f)
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.index<float>(i), output_cu.index<float>(i));
  }

  cudaStreamDestroy(stream);
}
#endif