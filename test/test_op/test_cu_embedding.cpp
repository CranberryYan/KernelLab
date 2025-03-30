#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../source/op/kernels/kernels_interface.h"

#if 0
TEST(test_embedding_cu, embedding_nostream) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token_num = 512;
  int32_t hidden_units = 128;
  int32_t vocab_size = 4096;
  int32_t size = token_num * hidden_units;

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32,
                           token_num, true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32,
                            token_num, hidden_units, true, alloc_cpu);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32,
                            vocab_size, hidden_units, true, alloc_cpu);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32,
                          token_num, true, alloc_cu, nullptr);
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
    input_cu.set_value<int32_t>(input_tmp, i);
  }

  for (int i = 0; i < vocab_size * hidden_units; ++i) {
    float weight_tmp = dist(mt);
    weight_cpu.set_value<float>(weight_tmp, i);
    weight_cu.set_value<float>(weight_tmp, i);
  }

  kernel::get_embedding_kernel(base::DeviceType::kDeviceCPU)(
    input_cpu, weight_cpu, output_cpu, vocab_size, nullptr);
  kernel::get_embedding_kernel(base::DeviceType::kDeviceCUDA)(
    input_cu, weight_cu, output_cu, vocab_size, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output_cpu.index<float>(i), output_cu.index<float>(i))
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.index<float>(i), output_cu.index<float>(i));
  }
}

TEST(test_embedding_cu, embedding_stream) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token_num = 512;
  int32_t hidden_units = 128;
  int32_t vocab_size = 4096;
  int32_t size = token_num * hidden_units;

  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32,
                           token_num, true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32,
                            token_num, hidden_units, true, alloc_cpu);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32,
                            vocab_size, hidden_units, true, alloc_cpu);
  tensor::Tensor input_cu(base::DataType::kDataTypeFp32,
                          token_num, true, alloc_cu, nullptr);
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
    input_cu.set_value<int32_t>(input_tmp, i);
  }

  for (int i = 0; i < vocab_size * hidden_units; ++i) {
    float weight_tmp = dist(mt);
    weight_cpu.set_value<float>(weight_tmp, i);
    weight_cu.set_value<float>(weight_tmp, i);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_embedding_kernel(base::DeviceType::kDeviceCPU)(
    input_cpu, weight_cpu, output_cpu, vocab_size, nullptr);
  kernel::get_embedding_kernel(base::DeviceType::kDeviceCUDA)(
    input_cu, weight_cu, output_cu, vocab_size, stream);

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output_cpu.index<float>(i), output_cu.index<float>(i))
      << printf("index: %d, CPU: %f, GPU: %f\n",
        i, output_cpu.index<float>(i), output_cu.index<float>(i));
  }
}
#endif
