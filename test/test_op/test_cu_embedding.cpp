#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/embedding.h"
#include "../source/op/kernels/kernels_interface.h"

#if 0
TEST(test_embedding_cu, embedding_stream) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token_num = 512;
  int32_t hidden_units = 128;
  int32_t vocab_size = 4096;
  int32_t size = token_num * hidden_units;

  tensor::Tensor input_token_num =
    tensor::Tensor(base::DataType::kDataTypeInt32, token_num);

  tensor::Tensor input_cpu(base::DataType::kDataTypeInt32,
                           token_num, true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32,
                            token_num, hidden_units, true, alloc_cpu);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32,
                            vocab_size, hidden_units, true, alloc_cpu);
  tensor::Tensor output_cu(base::DataType::kDataTypeFp32,
                           token_num, hidden_units, true, alloc_cu);
  tensor::Tensor weight_cu(base::DataType::kDataTypeFp32,
                           vocab_size, hidden_units, true, alloc_cu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 4096.f);
  for (int i = 0; i < token_num; ++i) {
    int32_t input_tmp = static_cast<int32_t>(dist(mt));
    input_cpu.set_value<int32_t>(input_tmp, i);
  }

  for (int i = 0; i < vocab_size * hidden_units; ++i) {
    float weight_tmp = dist(mt);
    weight_cpu.set_value<float>(weight_tmp, i);
    weight_cu.set_value<float>(weight_tmp, i);
  }

  std::shared_ptr<op::Layer> embedding_layer_cpu =
    std::make_shared<op::EmbeddingLayer>(base::DeviceType::kDeviceCPU,
      hidden_units, token_num, vocab_size);

  std::shared_ptr<op::Layer> embedding_layer_cu =
    std::make_shared<op::EmbeddingLayer>(base::DeviceType::kDeviceCUDA,
      hidden_units, token_num, vocab_size);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();

  config->stream = stream;
  embedding_layer_cu->set_cuda_config(config);

  embedding_layer_cpu->set_weight(0, {vocab_size, hidden_units},
                                  weight_cpu.ptr<float>(),
                                  base::DeviceType::kDeviceCPU);
  embedding_layer_cu->set_weight(0, {vocab_size, hidden_units},
                                 weight_cu.ptr<float>(),
                                 base::DeviceType::kDeviceCUDA);

  embedding_layer_cpu->forward(input_cpu, input_token_num, output_cpu);
  embedding_layer_cu->forward(input_cpu, input_token_num, output_cu);

  output_cu.to_cpu();

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output_cpu.at<float>(i), output_cu.at<float>(i))
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.at<float>(i), output_cu.at<float>(i));
  }
}
#endif
