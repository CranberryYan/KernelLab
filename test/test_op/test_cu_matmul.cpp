#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/matmul.h"
#include "../source/op/kernels/kernels_interface.h"

#if 0
TEST(test_matmul_cu, matmul_Sgemv_0) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 512, true, alloc_cpu);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32, 4096, 512, true, alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, 4096, true, alloc_cpu);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, 512, true, alloc_cu);
  tensor::Tensor weight_cu(base::DataType::kDataTypeFp32, 4096, 512, true, alloc_cu);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, 4096, true, alloc_cu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 10.f);
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

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;

  std::shared_ptr<op::Layer> matmul_layer_cpu =
    std::make_shared<op::MatmulLayer>(base::DeviceType::kDeviceCPU, 4096, 512);

  std::shared_ptr<op::Layer> matmul_layer_cu =
  std::make_shared<op::MatmulLayer>(base::DeviceType::kDeviceCUDA, 4096, 512);

  matmul_layer_cpu->set_weight(0, {4096, 512},
                               weight_cpu.ptr<float>(),
                               base::DeviceType::kDeviceCPU);
  matmul_layer_cu->set_weight(0, {4096, 512},
                              weight_cu.ptr<float>(),
                              base::DeviceType::kDeviceCUDA);

  matmul_layer_cu->set_cuda_config(config);

  matmul_layer_cpu->forward(input_cpu, output_cpu);
  matmul_layer_cu->forward(input_cu, output_cu);

  for (int i = 0; i < output_cpu.size(); ++i) {
    float diff = std::abs(
      output_cpu.at<float>(i) - output_cu.at<float>(i));
    if (diff > 1e-3) {
      printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
    }
  }
}
#endif
