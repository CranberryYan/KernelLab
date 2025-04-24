#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/scatter.h"
#include "../source/op/kernels/kernels_interface.h"

#if 1
TEST(test_cu_scatter, test_0_update) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int input_ele_num = 1;
  std::vector<uint32_t> intput_dims = {256, 1024};
  for (auto &dim : intput_dims) {
    input_ele_num *= dim;
  }

  int index_ele_num = 1;
  std::vector<uint32_t> index_dims = {256, 512};
  for (auto &dim : index_dims) {
    index_ele_num *= dim;
  }

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 256, 1024,
    true, alloc_cpu, nullptr);
  tensor::Tensor src_cpu(base::DataType::kDataTypeFp32, 256, 512,
    true, alloc_cpu, nullptr);
  tensor::Tensor index_cpu(base::DataType::kDataTypeInt32, 256, 512,
    true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, 256, 1024,
    true, alloc_cu, nullptr);
  tensor::Tensor src_cu(base::DataType::kDataTypeFp32, 256, 512,
    true, alloc_cu, nullptr);
  tensor::Tensor index_cu(base::DataType::kDataTypeInt32, 256, 512,
    true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(0.f, 1024.f);
  std::uniform_int_distribution<int> dist_int(0, 1023);
  for (int i = 0; i < input_ele_num; ++i) {
    float input_tmp = dist_float(mt);
    input_cpu.set_value<float>(input_tmp, i);
    input_cu.set_value<float>(input_tmp, i);
  }
  for (int i = 0; i < index_ele_num; ++i) {
    float src_tmp = dist_float(mt);
    int index_tmp = dist_int(mt);
    src_cpu.set_value<float>(src_tmp, i);
    src_cu.set_value<float>(src_tmp, i);
    index_cpu.set_value<int>(index_tmp, i);
    index_cu.set_value<int>(index_tmp, i);
  }

  std::shared_ptr<op::Layer> scatter_layer_cpu =
    std::make_shared<op::ScatterLayer>(base::DeviceType::kDeviceCPU);
  std::shared_ptr<op::Layer> scatter_layer_cu =
    std::make_shared<op::ScatterLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  scatter_layer_cu->set_cuda_config(config);

  // 下行转换, 访问派生类的成员变量
  std::shared_ptr<op::ScatterLayer> scatter_layer_cpu_ =
    std::dynamic_pointer_cast<op::ScatterLayer>(scatter_layer_cpu);
  scatter_layer_cpu_->op_type = para::ScatterOpType::Update;

  std::shared_ptr<op::ScatterLayer> scatter_layer_cu_ =
    std::dynamic_pointer_cast<op::ScatterLayer>(scatter_layer_cu);
  scatter_layer_cu_->op_type = para::ScatterOpType::Update;

  scatter_layer_cpu->forward(input_cpu, src_cpu, index_cpu, input_cpu);
  scatter_layer_cu->forward(input_cu, src_cu, index_cu, input_cu);

  input_cu.to_cpu();
  for (int i = 0; i < input_ele_num; ++i) {
    ASSERT_EQ(input_cpu.at<float>(i), input_cu.at<float>(i))
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, input_cpu.at<float>(i), input_cu.at<float>(i));
  }
}

TEST(test_cu_scatter, test_0_add) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int input_ele_num = 1;
  std::vector<uint32_t> intput_dims = {256, 1024};
  for (auto &dim : intput_dims) {
    input_ele_num *= dim;
  }

  int index_ele_num = 1;
  std::vector<uint32_t> index_dims = {256, 512};
  for (auto &dim : index_dims) {
    index_ele_num *= dim;
  }

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 256, 1024,
    true, alloc_cpu, nullptr);
  tensor::Tensor src_cpu(base::DataType::kDataTypeFp32, 256, 512,
    true, alloc_cpu, nullptr);
  tensor::Tensor index_cpu(base::DataType::kDataTypeInt32, 256, 512,
    true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, 256, 1024,
    true, alloc_cu, nullptr);
  tensor::Tensor src_cu(base::DataType::kDataTypeFp32, 256, 512,
    true, alloc_cu, nullptr);
  tensor::Tensor index_cu(base::DataType::kDataTypeInt32, 256, 512,
    true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(0.f, 1024.f);
  std::uniform_int_distribution<int> dist_int(0, 1023);
  for (int i = 0; i < input_ele_num; ++i) {
    float input_tmp = dist_float(mt);
    input_cpu.set_value<float>(input_tmp, i);
    input_cu.set_value<float>(input_tmp, i);
  }
  for (int i = 0; i < index_ele_num; ++i) {
    float src_tmp = dist_float(mt);
    int index_tmp = dist_int(mt);
    src_cpu.set_value<float>(src_tmp, i);
    src_cu.set_value<float>(src_tmp, i);
    index_cpu.set_value<int>(index_tmp, i);
    index_cu.set_value<int>(index_tmp, i);
  }

  std::shared_ptr<op::Layer> scatter_layer_cpu =
    std::make_shared<op::ScatterLayer>(base::DeviceType::kDeviceCPU);
  std::shared_ptr<op::Layer> scatter_layer_cu =
    std::make_shared<op::ScatterLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  scatter_layer_cu->set_cuda_config(config);

  // 下行转换, 访问派生类的成员变量
  std::shared_ptr<op::ScatterLayer> scatter_layer_cpu_ =
    std::dynamic_pointer_cast<op::ScatterLayer>(scatter_layer_cpu);
  scatter_layer_cpu_->op_type = para::ScatterOpType::Add;

  std::shared_ptr<op::ScatterLayer> scatter_layer_cu_ =
    std::dynamic_pointer_cast<op::ScatterLayer>(scatter_layer_cu);
  scatter_layer_cu_->op_type = para::ScatterOpType::Add;

  scatter_layer_cpu->forward(input_cpu, src_cpu, index_cpu, input_cpu);
  scatter_layer_cu->forward(input_cu, src_cu, index_cu, input_cu);

  input_cu.to_cpu();
  for (int i = 0; i < input_ele_num; ++i) {
    
    float diff = std::abs(
      input_cpu.at<float>(i) - input_cu.at<float>(i));
    if (diff > 1e-3) {
      printf("index: %d, CPU: %f, GPU: %f\n",
        i, input_cpu.at<float>(i), input_cu.at<float>(i));
    }
  }
}

TEST(test_cu_scatter, test_1_update) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int input_ele_num = 1;
  std::vector<uint32_t> intput_dims = {256, 1024};
  for (auto &dim : intput_dims) {
    input_ele_num *= dim;
  }

  int index_ele_num = 1;
  std::vector<uint32_t> index_dims = {256, 512};
  for (auto &dim : index_dims) {
    index_ele_num *= dim;
  }

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 256, 1024,
    true, alloc_cpu, nullptr);
  tensor::Tensor src_cpu(base::DataType::kDataTypeFp32, 256, 512,
    true, alloc_cpu, nullptr);
  tensor::Tensor index_cpu(base::DataType::kDataTypeInt32, 256, 512,
    true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, 256, 1024,
    true, alloc_cu, nullptr);
  tensor::Tensor src_cu(base::DataType::kDataTypeFp32, 256, 512,
    true, alloc_cu, nullptr);
  tensor::Tensor index_cu(base::DataType::kDataTypeInt32, 256, 512,
    true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(0.f, 1024.f);
  std::uniform_int_distribution<int> dist_int(0, 1023);
  for (int i = 0; i < input_ele_num; ++i) {
    float input_tmp = dist_float(mt);
    input_cpu.set_value<float>(input_tmp, i);
    input_cu.set_value<float>(input_tmp, i);
  }
  for (int i = 0; i < index_ele_num; ++i) {
    float src_tmp = dist_float(mt);
    int index_tmp = dist_int(mt);
    src_cpu.set_value<float>(src_tmp, i);
    src_cu.set_value<float>(src_tmp, i);
    index_cpu.set_value<int>(-index_tmp, i);
    index_cu.set_value<int>(-index_tmp, i);
  }

  std::shared_ptr<op::Layer> scatter_layer_cpu =
    std::make_shared<op::ScatterLayer>(base::DeviceType::kDeviceCPU);
  std::shared_ptr<op::Layer> scatter_layer_cu =
    std::make_shared<op::ScatterLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  scatter_layer_cu->set_cuda_config(config);

  // 下行转换, 访问派生类的成员变量
  std::shared_ptr<op::ScatterLayer> scatter_layer_cpu_ =
    std::dynamic_pointer_cast<op::ScatterLayer>(scatter_layer_cpu);
  scatter_layer_cpu_->op_type = para::ScatterOpType::Update;

  std::shared_ptr<op::ScatterLayer> scatter_layer_cu_ =
    std::dynamic_pointer_cast<op::ScatterLayer>(scatter_layer_cu);
  scatter_layer_cu_->op_type = para::ScatterOpType::Update;

  scatter_layer_cpu->forward(input_cpu, src_cpu, index_cpu, input_cpu);
  scatter_layer_cu->forward(input_cu, src_cu, index_cu, input_cu);

  input_cu.to_cpu();
  for (int i = 0; i < input_ele_num; ++i) {
    ASSERT_EQ(input_cpu.at<float>(i), input_cu.at<float>(i))
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, input_cpu.at<float>(i), input_cu.at<float>(i));
  }
}

TEST(test_cu_scatter, test_1_add) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int input_ele_num = 1;
  std::vector<uint32_t> intput_dims = {256, 1024};
  for (auto &dim : intput_dims) {
    input_ele_num *= dim;
  }

  int index_ele_num = 1;
  std::vector<uint32_t> index_dims = {256, 512};
  for (auto &dim : index_dims) {
    index_ele_num *= dim;
  }

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 256, 1024,
    true, alloc_cpu, nullptr);
  tensor::Tensor src_cpu(base::DataType::kDataTypeFp32, 256, 512,
    true, alloc_cpu, nullptr);
  tensor::Tensor index_cpu(base::DataType::kDataTypeInt32, 256, 512,
    true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, 256, 1024,
    true, alloc_cu, nullptr);
  tensor::Tensor src_cu(base::DataType::kDataTypeFp32, 256, 512,
    true, alloc_cu, nullptr);
  tensor::Tensor index_cu(base::DataType::kDataTypeInt32, 256, 512,
    true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(0.f, 1024.f);
  std::uniform_int_distribution<int> dist_int(0, 1023);
  for (int i = 0; i < input_ele_num; ++i) {
    float input_tmp = dist_float(mt);
    input_cpu.set_value<float>(input_tmp, i);
    input_cu.set_value<float>(input_tmp, i);
  }
  for (int i = 0; i < index_ele_num; ++i) {
    float src_tmp = dist_float(mt);
    int index_tmp = dist_int(mt);
    src_cpu.set_value<float>(src_tmp, i);
    src_cu.set_value<float>(src_tmp, i);
    index_cpu.set_value<int>(-index_tmp, i);
    index_cu.set_value<int>(-index_tmp, i);
  }

  std::shared_ptr<op::Layer> scatter_layer_cpu =
    std::make_shared<op::ScatterLayer>(base::DeviceType::kDeviceCPU);
  std::shared_ptr<op::Layer> scatter_layer_cu =
    std::make_shared<op::ScatterLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  scatter_layer_cu->set_cuda_config(config);

  // 下行转换, 访问派生类的成员变量
  std::shared_ptr<op::ScatterLayer> scatter_layer_cpu_ =
    std::dynamic_pointer_cast<op::ScatterLayer>(scatter_layer_cpu);
  scatter_layer_cpu_->op_type = para::ScatterOpType::Add;

  std::shared_ptr<op::ScatterLayer> scatter_layer_cu_ =
    std::dynamic_pointer_cast<op::ScatterLayer>(scatter_layer_cu);
  scatter_layer_cu_->op_type = para::ScatterOpType::Add;

  scatter_layer_cpu->forward(input_cpu, src_cpu, index_cpu, input_cpu);
  scatter_layer_cu->forward(input_cu, src_cu, index_cu, input_cu);

  input_cu.to_cpu();
  for (int i = 0; i < input_ele_num; ++i) {
    
    float diff = std::abs(
      input_cpu.at<float>(i) - input_cu.at<float>(i));
    if (diff > 1e-3) {
      printf("index: %d, CPU: %f, GPU: %f\n",
        i, input_cpu.at<float>(i), input_cu.at<float>(i));
    }
  }
}
#endif
