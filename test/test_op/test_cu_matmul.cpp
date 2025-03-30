#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../source/op/kernels/kernels_interface.h"

TEST(test_matmul_cu, matmul_Sgemv) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 512, true, alloc_cpu);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32, 4096, 512, true, alloc_cpu);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, 4096, true, alloc_cu);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, 512, true, alloc_cu);
  tensor::Tensor weight_cu(base::DataType::kDataTypeFp32, 4096, 512, true, alloc_cu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, 4096, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < 512; ++i) {
    float input_tmp = static_cast<float>(dist(mt));
    input_cpu.set_value(input_tmp, i);
    input_cu.set_value(input_tmp, i);
  }

  for (int i = 0; i < 512 * 4096; ++i) {
    float weight_tmp = static_cast<float>(dist(mt));
    weight_cpu.set_value(weight_tmp, i);
    weight_cu.set_value(weight_tmp, i);
  }

  kernel::CudaConfig* config = new kernel::CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input_cu,
                                                           weight_cu,
                                                           output_cu,
                                                           1.f,
                                                           config);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu,
                                                          weight_cpu,
                                                          output_cpu,
                                                          1.f,
                                                          config);

  for (int i = 0; i < output_cpu.size(); ++i) {
    float diff = std::abs(
      output_cpu.index<float>(i) - output_cu.index<float>(i));
    if (diff > 1e-3) {
      printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.index<float>(i), output_cu.index<float>(i));
    }
  }
}

TEST(test_matmul_cu, matmul_Sgemv_1) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 128, true, alloc_cpu);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32, 2048, 128, true, alloc_cpu);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, 2048, true, alloc_cu);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, 128, true, alloc_cu);
  tensor::Tensor weight_cu(base::DataType::kDataTypeFp32, 2048, 128, true, alloc_cu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, 2048, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < 512; ++i) {
    float input_tmp = static_cast<float>(dist(mt));
    input_cpu.set_value(input_tmp, i);
    input_cu.set_value(input_tmp, i);
  }

  for (int i = 0; i < 128 * 2048; ++i) {
    float weight_tmp = static_cast<float>(dist(mt));
    weight_cpu.set_value(weight_tmp, i);
    weight_cu.set_value(weight_tmp, i);
  }

  kernel::CudaConfig* config = new kernel::CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input_cu,
                                                           weight_cu,
                                                           output_cu,
                                                           1.f,
                                                           config);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu,
                                                          weight_cpu,
                                                          output_cpu,
                                                          1.f,
                                                          config);

  for (int i = 0; i < output_cpu.size(); ++i) {
    float diff = std::abs(
      output_cpu.index<float>(i) - output_cu.index<float>(i));
    if (diff > 1e-3) {
      printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.index<float>(i), output_cu.index<float>(i));
    }
  }
}