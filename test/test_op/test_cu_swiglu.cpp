#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/swiglu.h"
#include "../source/op/kernels/kernels_interface.h"

#if 0
TEST(test_swiglu_cu, swiglu_stream) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();

  int32_t hidden_units = 4096;

  tensor::Tensor input1_cu(base::DataType::kDataTypeFp32,
                           hidden_units, true, alloc_cu);
  tensor::Tensor input2_cu(base::DataType::kDataTypeFp32,
                           hidden_units, true, alloc_cu);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32,
                           hidden_units, true, alloc_cu);
  tensor::Tensor input1_cpu(base::DataType::kDataTypeFp32,
                            hidden_units, true, alloc_cpu);
  tensor::Tensor input2_cpu(base::DataType::kDataTypeFp32,
                            hidden_units, true, alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32,
                            hidden_units, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < hidden_units; ++i) {
    float input_tmp = dist(mt);
    input1_cu.set_value(input_tmp, i);
    input1_cpu.set_value(input_tmp, i);
    input2_cu.set_value(input_tmp, i);
    input2_cpu.set_value(input_tmp, i);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;

  std::shared_ptr<op::Layer> swiglu_layer_cpu =
    std::make_shared<op::SwiGLULayer>(base::DeviceType::kDeviceCPU,
                                      hidden_units);

  std::shared_ptr<op::Layer> swiglu_layer_cu =
  std::make_shared<op::SwiGLULayer>(base::DeviceType::kDeviceCUDA,
                                    hidden_units);

  swiglu_layer_cu->set_cuda_config(config);

  swiglu_layer_cpu->forward(input1_cpu, input2_cpu, output_cpu);
  swiglu_layer_cu->forward(input1_cu, input2_cu, output_cu);

  output_cu.to_cpu();

  for (int i = 0; i < hidden_units; ++i) {
    ASSERT_NEAR(output_cu.at<float>(i), output_cpu.at<float>(i), 1e-5f)
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
  }
}
#endif
