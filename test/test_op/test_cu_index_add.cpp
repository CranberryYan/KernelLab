#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/index_add.h"
#include "../source/op/kernels/kernels_interface.h"

#if 1
// shape与deepseek对齐
//  target: cols为7168   rows: {2, 250, 1024}
//  index: rows * 8 -> {16, 4000, 8192}
//  index: 一维向量, 值域: [0, target_rows)
//  source: shape: [index.size(), self_cols]
TEST(test_cu_index_add, test_GCU300_0) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  uint32_t target_rows = 2;
  uint32_t target_cols = 7168;
  uint32_t index_rows = 2 * 8;
  uint32_t source_rows = index_rows;
  uint32_t source_cols = target_cols;
  uint32_t target_ele_num = target_rows * target_cols;
  uint32_t index_ele_num = index_rows;
  uint32_t source_ele_num = source_rows * source_cols;
  std::vector<uint32_t> target_dims = {target_rows, target_cols};
  std::vector<uint32_t> index_dims = {index_rows};
  std::vector<uint32_t> source_dims = {source_rows, source_cols};

  tensor::Tensor target_cpu(base::DataType::kDataTypeFp32,
                            target_rows, target_cols,
                            true, alloc_cpu, nullptr);
  tensor::Tensor index_cpu(base::DataType::kDataTypeInt32,
                           index_rows,
                           true, alloc_cpu, nullptr);
  tensor::Tensor source_cpu(base::DataType::kDataTypeFp32,
                            source_rows, source_cols,
                            true, alloc_cpu, nullptr);

  tensor::Tensor target_cu(base::DataType::kDataTypeFp32,
                            target_rows, target_cols,
                            true, alloc_cu, nullptr);
  tensor::Tensor index_cu(base::DataType::kDataTypeInt32,
                           index_rows,
                           true, alloc_cu, nullptr);
  tensor::Tensor source_cu(base::DataType::kDataTypeFp32,
                            source_rows, source_cols,
                            true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> target_source_num(-100.0f, 100.0f);
  std::uniform_int_distribution<int32_t> index_num(0, target_rows - 1);
  for (int i = 0; i < target_ele_num; ++i) {
    float tmp = target_source_num(mt);
    // float tmp = 1.0f;
    target_cpu.set_value<float>(tmp, i);
  }
  target_cu = target_cpu.clone();
  target_cu.to_cuda();
  for (int i = 0; i < source_ele_num; ++i) {
    // float tmp = target_source_num(mt);
    float tmp = 1.0f;
    source_cpu.set_value<float>(tmp, i);
  }
  source_cu = source_cpu.clone();
  source_cu.to_cuda();
  for (int i = 0; i < index_ele_num; ++i) {
    int32_t tmp = index_num(mt);
    // float tmp = 1.0f;
    index_cpu.set_value<int32_t>(tmp, i);
    index_cu.set_value<int32_t>(tmp, i);
  }

  std::shared_ptr<op::Layer> index_add_layer_cpu =
    std::make_shared<op::IndexAddLayer>(base::DeviceType::kDeviceCPU);
  std::shared_ptr<op::Layer> index_add_layer_cu =
    std::make_shared<op::IndexAddLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  index_add_layer_cu->set_cuda_config(config);

  std::shared_ptr<op::IndexAddLayer> index_add_layer_cpu_ =
    std::dynamic_pointer_cast<op::IndexAddLayer>(index_add_layer_cpu);
  index_add_layer_cpu_->enflame_device = para::EnflameDevice::GCU300;

  std::shared_ptr<op::IndexAddLayer> index_add_layer_cu_ =
    std::dynamic_pointer_cast<op::IndexAddLayer>(index_add_layer_cu);
  index_add_layer_cu_->enflame_device = para::EnflameDevice::GCU300;

  index_add_layer_cpu->forward(target_cpu, index_cpu, source_cpu, target_cpu);
  index_add_layer_cu->forward(target_cu, index_cu, source_cu, target_cu);

  target_cu.to_cpu();
  uint32_t err_num = 0;
  for (int i = 0; i < target_ele_num; ++i) {
    if (i < 10) {
      printf("index: %d, target_cpu: %f, target_cu: %f\n",
        i, target_cpu.at<float>(i), target_cu.at<float>(i));
    }
    float diff = std::abs(
      target_cpu.at<float>(i) - target_cu.at<float>(i));
    if (diff > 1e-5) {
      err_num++;
      if (err_num > 20) {
        break;
      }
      printf("index: %d, target_cpu: %f, target_cu: %f\n",
        i, target_cpu.at<float>(i), target_cu.at<float>(i));
    }
  }
}

TEST(test_cu_index_add, test_GCU300_1) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  uint32_t target_rows = 512;
  uint32_t target_cols = 7168;
  uint32_t index_rows = 512 * 8;
  uint32_t source_rows = index_rows;
  uint32_t source_cols = target_cols;
  uint32_t target_ele_num = target_rows * target_cols;
  uint32_t index_ele_num = index_rows;
  uint32_t source_ele_num = source_rows * source_cols;
  std::vector<uint32_t> target_dims = {target_rows, target_cols};
  std::vector<uint32_t> index_dims = {index_rows};
  std::vector<uint32_t> source_dims = {source_rows, source_cols};

  tensor::Tensor target_cpu(base::DataType::kDataTypeFp32,
                            target_rows, target_cols,
                            true, alloc_cpu, nullptr);
  tensor::Tensor index_cpu(base::DataType::kDataTypeInt32,
                           index_rows,
                           true, alloc_cpu, nullptr);
  tensor::Tensor source_cpu(base::DataType::kDataTypeFp32,
                            source_rows, source_cols,
                            true, alloc_cpu, nullptr);

  tensor::Tensor target_cu(base::DataType::kDataTypeFp32,
                            target_rows, target_cols,
                            true, alloc_cu, nullptr);
  tensor::Tensor index_cu(base::DataType::kDataTypeInt32,
                           index_rows,
                           true, alloc_cu, nullptr);
  tensor::Tensor source_cu(base::DataType::kDataTypeFp32,
                            source_rows, source_cols,
                            true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> target_source_num(-100.0f, 100.0f);
  std::uniform_int_distribution<int32_t> index_num(0, target_rows - 1);
  for (int i = 0; i < target_ele_num; ++i) {
    float tmp = target_source_num(mt);
    target_cpu.set_value<float>(tmp, i);
  }
  target_cu = target_cpu.clone();
  target_cu.to_cuda();
  for (int i = 0; i < source_ele_num; ++i) {
    float tmp = target_source_num(mt);
    source_cpu.set_value<float>(tmp, i);
  }
  source_cu = source_cpu.clone();
  source_cu.to_cuda();
  for (int i = 0; i < index_ele_num; ++i) {
    int32_t tmp = index_num(mt);
    index_cpu.set_value<int32_t>(tmp, i);
    index_cu.set_value<int32_t>(tmp, i);
  }

  for (int i = 0; i < 128; ++i) {
    printf(" ========== before: cpu[%d]: %f source[%d]: %f\n",
      i, target_cpu.at<float>(i), i, source_cpu.at<float>(i));
  }

  std::shared_ptr<op::Layer> index_add_layer_cpu =
    std::make_shared<op::IndexAddLayer>(base::DeviceType::kDeviceCPU);
  std::shared_ptr<op::Layer> index_add_layer_cu =
    std::make_shared<op::IndexAddLayer>(base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  index_add_layer_cu->set_cuda_config(config);

  std::shared_ptr<op::IndexAddLayer> index_add_layer_cpu_ =
    std::dynamic_pointer_cast<op::IndexAddLayer>(index_add_layer_cpu);
  index_add_layer_cpu_->enflame_device = para::EnflameDevice::GCU300;

  std::shared_ptr<op::IndexAddLayer> index_add_layer_cu_ =
    std::dynamic_pointer_cast<op::IndexAddLayer>(index_add_layer_cu);
  index_add_layer_cu_->enflame_device = para::EnflameDevice::GCU300;

  index_add_layer_cpu->forward(target_cpu, index_cpu, source_cpu, target_cpu);
  index_add_layer_cu->forward(target_cu, index_cu, source_cu, target_cu);

  target_cu.to_cpu();
  uint32_t err_num = 0;
  for (int i = 0; i < target_ele_num; ++i) {
    if (i < 10) {
      printf("index: %d, target_cpu: %f, target_cu: %f\n",
        i, target_cpu.at<float>(i), target_cu.at<float>(i));
    }
    float diff = std::abs(
      target_cpu.at<float>(i) - target_cu.at<float>(i));
    if (diff > 1e-5) {
      err_num++;
      if (err_num > 20) {
        break;
      }
      printf("index: %d, target_cpu: %f, target_cu: %f\n",
        i, target_cpu.at<float>(i), target_cu.at<float>(i));
    }
  }
}
#endif
