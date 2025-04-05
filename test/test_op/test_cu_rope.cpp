#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../source/op/kernels/cpu/rope_kernel.h"
#include "../source/op/kernels/cuda/rope_kernel.cuh"
#include "../source/op/kernels/kernels_interface.h"

TEST(test_rope_cu, rope_nostream) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();

  int32_t dim = 256;
  int32_t head_num = 4;
  int32_t head_size = 64;
  int32_t kv_dim = 128;
  int32_t pos = 3;
  int32_t max_seq_len = 2048;
  tensor::Tensor input_pos(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  input_pos.index<int32_t>(0) = pos;

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  tensor::Tensor input_q_cpu(base::DataType::kDataTypeFp32, dim,
                             true, alloc_cpu, nullptr);
  tensor::Tensor input_k_cpu(base::DataType::kDataTypeFp32, dim,
                             true, alloc_cpu, nullptr);
  tensor::Tensor sin_cache_cpu(base::DataType::kDataTypeFp32, max_seq_len, dim,
                               true, alloc_cpu, nullptr);
  tensor::Tensor cos_cache_cpu(base::DataType::kDataTypeFp32, max_seq_len, dim,
                               true, alloc_cpu, nullptr);

  tensor::Tensor input_q_cu(base::DataType::kDataTypeFp32, dim,
                            true, alloc_cu, nullptr);
  tensor::Tensor input_k_cu(base::DataType::kDataTypeFp32, dim,
                            true, alloc_cu, nullptr);
  tensor::Tensor sin_cache_cu(base::DataType::kDataTypeFp32, max_seq_len, dim,
                              true, alloc_cu, nullptr);
  tensor::Tensor cos_cache_cu(base::DataType::kDataTypeFp32, max_seq_len, dim,
                              true, alloc_cu, nullptr);

  kernel::sin_cos_cache_calc_cpu(head_size, max_seq_len,
    sin_cache_cpu.ptr<float>(), cos_cache_cpu.ptr<float>());

  kernel::sin_cos_cache_calc_cu(head_size, max_seq_len,
    sin_cache_cu, cos_cache_cu, nullptr);

  for (int i = 0; i < dim; ++i) {
    float q = dist(mt);
    float k = dist(mt);
    input_q_cpu.set_value(q, i);
    input_k_cpu.set_value(k, i);
    input_q_cu.set_value(q, i);
    input_k_cu.set_value(k, i);
  }

  kernel::get_rope_kernel(base::DeviceType::kDeviceCPU)(
    dim, kv_dim, head_size, input_q_cpu, input_k_cpu, input_pos,
    sin_cache_cpu, cos_cache_cpu, nullptr);

  kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(
    dim, kv_dim, head_size, input_q_cu, input_k_cu, input_pos,
    sin_cache_cu, cos_cache_cu, nullptr);
  cudaDeviceSynchronize();

  for (int32_t i = 0; i < dim; ++i) {
    ASSERT_NEAR(input_k_cpu.index<float>(i), input_k_cu.index<float>(i), 1e-3f)
      << "ik: " << i;
    ASSERT_NEAR(input_q_cpu.index<float>(i), input_q_cu.index<float>(i), 1e-3f)
      << "iq: " << i;
  }
}

TEST(test_rope_cu, rope_stream) {
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();

  int32_t dim = 256;
  int32_t head_num = 4;
  int32_t head_size = 64;
  int32_t kv_dim = 128;
  int32_t pos = 3;
  int32_t max_seq_len = 2048;
  tensor::Tensor input_pos(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  input_pos.index<int32_t>(0) = pos;

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  tensor::Tensor input_q_cpu(base::DataType::kDataTypeFp32, dim,
                             true, alloc_cpu, nullptr);
  tensor::Tensor input_k_cpu(base::DataType::kDataTypeFp32, dim,
                             true, alloc_cpu, nullptr);
  tensor::Tensor sin_cache_cpu(base::DataType::kDataTypeFp32, max_seq_len, dim,
                               true, alloc_cpu, nullptr);
  tensor::Tensor cos_cache_cpu(base::DataType::kDataTypeFp32, max_seq_len, dim,
                               true, alloc_cpu, nullptr);

  tensor::Tensor input_q_cu(base::DataType::kDataTypeFp32, dim,
                            true, alloc_cu, nullptr);
  tensor::Tensor input_k_cu(base::DataType::kDataTypeFp32, dim,
                            true, alloc_cu, nullptr);
  tensor::Tensor sin_cache_cu(base::DataType::kDataTypeFp32, max_seq_len, dim,
                              true, alloc_cu, nullptr);
  tensor::Tensor cos_cache_cu(base::DataType::kDataTypeFp32, max_seq_len, dim,
                              true, alloc_cu, nullptr);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  kernel::sin_cos_cache_calc_cpu(head_size, max_seq_len,
    sin_cache_cpu.ptr<float>(), cos_cache_cpu.ptr<float>());

  kernel::sin_cos_cache_calc_cu(head_size, max_seq_len,
    sin_cache_cu, cos_cache_cu, stream);

  for (int i = 0; i < dim; ++i) {
    float q = dist(mt);
    float k = dist(mt);
    input_q_cpu.set_value(q, i);
    input_k_cpu.set_value(k, i);
    input_q_cu.set_value(q, i);
    input_k_cu.set_value(k, i);
  }

  kernel::get_rope_kernel(base::DeviceType::kDeviceCPU)(
    dim, kv_dim, head_size, input_q_cpu, input_k_cpu, input_pos,
    sin_cache_cpu, cos_cache_cpu, nullptr);

  kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(
    dim, kv_dim, head_size, input_q_cu, input_k_cu, input_pos,
    sin_cache_cu, cos_cache_cu, stream);
  cudaDeviceSynchronize();

  for (int32_t i = 0; i < dim; ++i) {
    ASSERT_NEAR(input_k_cpu.index<float>(i), input_k_cu.index<float>(i), 1e-3f)
      << "ik: " << i;
    ASSERT_NEAR(input_q_cpu.index<float>(i), input_q_cu.index<float>(i), 1e-3f)
      << "iq: " << i;
  }
}
