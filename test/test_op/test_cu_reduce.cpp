#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/reduce.h"
#include "../source/op/kernels/kernels_interface.h"

#if 0
TEST(test_cu_reduce, test_0) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int32_t after_reduce_num = 127;
  int32_t ele_num = 1;
  std::vector<int32_t> dims = {127, 1024};
  for (auto& dim : dims) {
    ele_num *= dim;
  }

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, after_reduce_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, after_reduce_num,
                         true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < ele_num; ++i) {
    float input_tmp = dist(mt);
    input_cpu.set_value<float>(input_tmp, i);
    input_cu.set_value<float>(input_tmp, i);
  }

  std::shared_ptr<op::Layer> reduce_layer_cpu =
    std::make_shared<op::ReduceLayer>(base::DeviceType::kDeviceCPU);

  std::shared_ptr<op::Layer> reduce_layer_cu =
    std::make_shared<op::ReduceLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  reduce_layer_cu->set_cuda_config(config);

  reduce_layer_cpu->forward(input_cpu, output_cpu);
  reduce_layer_cu->forward(input_cu, output_cu);

  output_cu.to_cpu();
  for (int i = 0; i < after_reduce_num; ++i) {
    float diff = 1e-3;
    if (std::abs(output_cpu.at<float>(i)) -
        std::abs(output_cu.at<float>(i)) > diff) {
      printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
    }
  }
}
#endif
