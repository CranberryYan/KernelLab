#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/add.h"
#include "../source/op/kernels/kernels_interface.h"

#if 0
TEST(test_cu_add, test_0) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int32_t ele_num = 1;
  std::vector<int32_t> dims = {128, 2048};
  for (auto& dim : dims) {
    ele_num *= dim;
  }

  tensor::Tensor input1_cpu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input2_cpu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input1_cu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cu, nullptr);
  tensor::Tensor input2_cu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < ele_num; ++i) {
    float input1_tmp = dist(mt);
    float input2_tmp = dist(mt);
    input1_cpu.set_value<float>(input1_tmp, i);
    input2_cpu.set_value<float>(input2_tmp, i);
    input1_cu.set_value<float>(input1_tmp, i);
    input2_cu.set_value<float>(input2_tmp, i);
  }

  std::shared_ptr<op::Layer> add_layer_cpu =
    std::make_shared<op::VecAddLayer>(base::DeviceType::kDeviceCPU);

  std::shared_ptr<op::Layer> add_layer_cu =
    std::make_shared<op::VecAddLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  add_layer_cu->set_cuda_config(config);

  add_layer_cpu->forward(input1_cpu, input2_cpu, output_cpu);
  add_layer_cu->forward(input1_cu, input2_cu, output_cu);

  output_cu.to_cpu();
  for (int i = 0; i < ele_num; ++i) {
    ASSERT_EQ(output_cpu.at<float>(i), output_cu.at<float>(i))
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
  }
}

TEST(test_cu_add, test_1) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int32_t ele_num = 1;
  std::vector<int32_t> dims = {256, 4096};
  for (auto& dim : dims) {
    ele_num *= dim;
  }

  tensor::Tensor input1_cpu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input2_cpu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input1_cu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cu, nullptr);
  tensor::Tensor input2_cu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < ele_num; ++i) {
    float input1_tmp = dist(mt);
    float input2_tmp = dist(mt);
    input1_cpu.set_value<float>(input1_tmp, i);
    input2_cpu.set_value<float>(input2_tmp, i);
    input1_cu.set_value<float>(input1_tmp, i);
    input2_cu.set_value<float>(input2_tmp, i);
  }

  std::shared_ptr<op::Layer> add_layer_cpu =
    std::make_shared<op::VecAddLayer>(base::DeviceType::kDeviceCPU);

  std::shared_ptr<op::Layer> add_layer_cu =
    std::make_shared<op::VecAddLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  add_layer_cu->set_cuda_config(config);

  add_layer_cpu->forward(
    input1_cpu, input2_cpu, output_cpu);
  add_layer_cu->forward(
    input1_cu, input2_cu, output_cu);

  output_cu.to_cpu();
  for (int i = 0; i < ele_num; ++i) {
    ASSERT_EQ(output_cpu.at<float>(i), output_cu.at<float>(i))
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
  }
}

TEST(test_cu_add, test_2) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int32_t ele_num = 1;
  std::vector<int32_t> dims = {256 + 1, 4096 + 1};
  for (auto& dim : dims) {
    ele_num *= dim;
  }

  tensor::Tensor input1_cpu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input2_cpu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cpu, nullptr);
  tensor::Tensor input1_cu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cu, nullptr);
  tensor::Tensor input2_cu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, ele_num,
                         true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < ele_num; ++i) {
    float input1_tmp = dist(mt);
    float input2_tmp = dist(mt);
    input1_cpu.set_value<float>(input1_tmp, i);
    input2_cpu.set_value<float>(input2_tmp, i);
    input1_cu.set_value<float>(input1_tmp, i);
    input2_cu.set_value<float>(input2_tmp, i);
  }

  std::shared_ptr<op::Layer> add_layer_cpu =
    std::make_shared<op::VecAddLayer>(base::DeviceType::kDeviceCPU);

  std::shared_ptr<op::Layer> add_layer_cu =
    std::make_shared<op::VecAddLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  add_layer_cu->set_cuda_config(config);

  add_layer_cpu->forward(
    input1_cpu, input2_cpu, output_cpu);
  add_layer_cu->forward(
    input1_cu, input2_cu, output_cu);

  output_cu.to_cpu();
  for (int i = 0; i < ele_num; ++i) {
    ASSERT_EQ(output_cpu.at<float>(i), output_cu.at<float>(i))
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
  }
}
#endif
