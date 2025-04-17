#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/rmsnorm.h"
#include "../source/op/kernels/kernels_interface.h"

#if 0
TEST(test_rmsnorm_cu, rmsnorm_stream) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();

  int32_t total_num = 256 * 4096;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;

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

  std::shared_ptr<op::Layer> rmsnorm_layer_cpu =
    std::make_shared<op::RMSNormLayer>(base::DeviceType::kDeviceCPU,
                                       total_num);

  std::shared_ptr<op::Layer> rmsnorm_layer_cu =
    std::make_shared<op::RMSNormLayer>(base::DeviceType::kDeviceCUDA,
                                       total_num);

  rmsnorm_layer_cpu->set_weight(0, {total_num},
    weight_cpu.ptr<float>(), base::DeviceType::kDeviceCPU);

  rmsnorm_layer_cu->set_weight(0, {total_num},
    weight_cu.ptr<float>(), base::DeviceType::kDeviceCUDA);
  rmsnorm_layer_cu->set_cuda_config(config);

  rmsnorm_layer_cpu->forward(input_cpu, output_cpu);
  rmsnorm_layer_cu->forward(input_cu, output_cu);

  output_cu.to_cpu();

  for (int i = 0; i < total_num; ++i) {
    ASSERT_NEAR(output_cu.at<float>(i), output_cpu.at<float>(i), 1e-3f)
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
  }

  cudaStreamDestroy(stream);
}
#endif
