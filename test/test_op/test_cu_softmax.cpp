#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/softmax.h"
#include "../source/op/kernels/kernels_interface.h"

#if 0
TEST(test_cpu_softmax, test_0_safe_and_online) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();

  int input_ele_num = 1;
  std::vector<uint32_t> intput_dims = {4096, 128};
  for (auto &dim : intput_dims) {
    input_ele_num *= dim;
  }

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu_safe(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu_online(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cpu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(-5.0f, 10.f);
  for (int i = 0; i < input_ele_num; ++i) {
    float input_tmp = dist_float(mt);
    input_cpu.set_value<float>(input_tmp, i);
  }

  std::shared_ptr<op::Layer> softmax_layer_cpu_safe =
    std::make_shared<op::SoftmaxLayer>(base::DeviceType::kDeviceCPU);
  std::shared_ptr<op::Layer> softmax_layer_cpu_online =
    std::make_shared<op::SoftmaxLayer>(base::DeviceType::kDeviceCPU);

  // 下行转换, 访问派生类的成员变量
  std::shared_ptr<op::SoftmaxLayer> softmax_layer_cpu_safe_ =
    std::dynamic_pointer_cast<op::SoftmaxLayer>(softmax_layer_cpu_safe);
  softmax_layer_cpu_safe_->op_type = para::SoftmaxOpType::Safe;

  std::shared_ptr<op::SoftmaxLayer> softmax_layer_cpu_online_ =
    std::dynamic_pointer_cast<op::SoftmaxLayer>(softmax_layer_cpu_online);
  softmax_layer_cpu_online_->op_type = para::SoftmaxOpType::Online;

  softmax_layer_cpu_safe->forward(input_cpu, output_cpu_safe);
  softmax_layer_cpu_online->forward(input_cpu, output_cpu_online);

  for (int i = 0; i < input_ele_num; ++i) {
    float diff = std::abs(
      output_cpu_safe.at<float>(i) - output_cpu_online.at<float>(i));
    if (diff > 1e-5) {
      printf("index: %d, output_cpu_safe: %f, output_cpu_online: %f\n",
        i, output_cpu_safe.at<float>(i), output_cpu_online.at<float>(i));
    }
  }
}

TEST(test_cu_softmax, test_0_naive) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int input_ele_num = 1;
  std::vector<uint32_t> intput_dims = {4096, 128};
  for (auto &dim : intput_dims) {
    input_ele_num *= dim;
  }

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(-5.f, 5.f);
  for (int i = 0; i < input_ele_num; ++i) {
    float input_tmp = dist_float(mt);
    input_cpu.set_value<float>(input_tmp, i);
    input_cu.set_value<float>(input_tmp, i);
  }

  std::shared_ptr<op::Layer> softmax_layer_cpu =
    std::make_shared<op::SoftmaxLayer>(base::DeviceType::kDeviceCPU);
  std::shared_ptr<op::Layer> softmax_layer_cu =
    std::make_shared<op::SoftmaxLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  softmax_layer_cu->set_cuda_config(config);

  // 下行转换, 访问派生类的成员变量
  std::shared_ptr<op::SoftmaxLayer> softmax_layer_cpu_ =
    std::dynamic_pointer_cast<op::SoftmaxLayer>(softmax_layer_cpu);
  softmax_layer_cpu_->op_type = para::SoftmaxOpType::Naive;

  std::shared_ptr<op::SoftmaxLayer> softmax_layer_cu_ =
    std::dynamic_pointer_cast<op::SoftmaxLayer>(softmax_layer_cu);
  softmax_layer_cu_->op_type = para::SoftmaxOpType::Naive;

  softmax_layer_cpu->forward(input_cpu, output_cpu);
  softmax_layer_cu->forward(input_cu, output_cu);

  output_cu.to_cpu();
  uint32_t err_num = 0;
  for (int i = 0; i < input_ele_num; ++i) {
    if (i < 10) {
      printf("index: %d, output_cpu: %f, output_cu: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
    }
    float diff = std::abs(
      output_cpu.at<float>(i) - output_cu.at<float>(i));
    if (diff > 1e-5) {
      err_num++;
      if (err_num > 20) {
        break;
      }
      printf("index: %d, output_cpu: %f, output_cu: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
    }
  }
}

TEST(test_cu_softmax, test_0_safe) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int input_ele_num = 1;
  std::vector<uint32_t> intput_dims = {4096, 128};
  for (auto &dim : intput_dims) {
    input_ele_num *= dim;
  }

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(-5.f, 5.f);
  for (int i = 0; i < input_ele_num; ++i) {
    float input_tmp = dist_float(mt);
    input_cpu.set_value<float>(input_tmp, i);
    input_cu.set_value<float>(input_tmp, i);
  }

  std::shared_ptr<op::Layer> softmax_layer_cpu =
    std::make_shared<op::SoftmaxLayer>(base::DeviceType::kDeviceCPU);
  std::shared_ptr<op::Layer> softmax_layer_cu =
    std::make_shared<op::SoftmaxLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  softmax_layer_cu->set_cuda_config(config);

  // 下行转换, 访问派生类的成员变量
  std::shared_ptr<op::SoftmaxLayer> softmax_layer_cpu_ =
    std::dynamic_pointer_cast<op::SoftmaxLayer>(softmax_layer_cpu);
  softmax_layer_cpu_->op_type = para::SoftmaxOpType::Safe;

  std::shared_ptr<op::SoftmaxLayer> softmax_layer_cu_ =
    std::dynamic_pointer_cast<op::SoftmaxLayer>(softmax_layer_cu);
  softmax_layer_cu_->op_type = para::SoftmaxOpType::Safe;

  softmax_layer_cpu->forward(input_cpu, output_cpu);
  softmax_layer_cu->forward(input_cu, output_cu);

  output_cu.to_cpu();
  uint32_t err_num = 0;
  for (int i = 0; i < input_ele_num; ++i) {
    if (i < 10) {
      printf("index: %d, output_cpu: %f, output_cu: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
    }
    float diff = std::abs(
      output_cpu.at<float>(i) - output_cu.at<float>(i));
    if (diff > 1e-5) {
      err_num++;
      if (err_num > 20) {
        break;
      }
      printf("index: %d, output_cpu: %f, output_cu: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
    }
  }
}

TEST(test_cu_softmax, test_0_online) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int input_ele_num = 1;
  std::vector<uint32_t> intput_dims = {4096, 128};
  for (auto &dim : intput_dims) {
    input_ele_num *= dim;
  }

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, 4096, 128,
    true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(-5.f, 5.f);
  for (int i = 0; i < input_ele_num; ++i) {
    float input_tmp = dist_float(mt);
    input_cpu.set_value<float>(input_tmp, i);
    input_cu.set_value<float>(input_tmp, i);
  }

  std::shared_ptr<op::Layer> softmax_layer_cpu =
    std::make_shared<op::SoftmaxLayer>(base::DeviceType::kDeviceCPU);
  std::shared_ptr<op::Layer> softmax_layer_cu =
    std::make_shared<op::SoftmaxLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  softmax_layer_cu->set_cuda_config(config);

  // 下行转换, 访问派生类的成员变量
  std::shared_ptr<op::SoftmaxLayer> softmax_layer_cpu_ =
    std::dynamic_pointer_cast<op::SoftmaxLayer>(softmax_layer_cpu);
  softmax_layer_cpu_->op_type = para::SoftmaxOpType::Online;

  std::shared_ptr<op::SoftmaxLayer> softmax_layer_cu_ =
    std::dynamic_pointer_cast<op::SoftmaxLayer>(softmax_layer_cu);
  softmax_layer_cu_->op_type = para::SoftmaxOpType::Online;

  softmax_layer_cpu->forward(input_cpu, output_cpu);
  softmax_layer_cu->forward(input_cu, output_cu);

  output_cu.to_cpu();
  uint32_t err_num = 0;
  for (int i = 0; i < input_ele_num; ++i) {
    if (i < 10) {
      printf("index: %d, output_cpu: %f, output_cu: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
    }
    float diff = std::abs(
      output_cpu.at<float>(i) - output_cu.at<float>(i));
    if (diff > 1e-5) {
      err_num++;
      if (err_num > 20) {
        break;
      }
      printf("index: %d, output_cpu: %f, output_cu: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
    }
  }
}
#endif