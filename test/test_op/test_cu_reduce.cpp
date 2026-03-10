#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/reduce.h"
#include "../source/op/kernels/kernels_interface.h"

#if 0
void reduce(unsigned int rows, unsigned int cols, unsigned int reduce_mode) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  unsigned int after_reduce_num = 1;
  int32_t ele_num = 1;
  std::vector<unsigned int> dims = {rows, cols};
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
  std::uniform_real_distribution<float> dist_float(-100.f, 100.f);
  for (int i = 0; i < ele_num; ++i) {
    float input_tmp = dist_float(mt);
    input_cpu.set_value<float>(input_tmp, i);
    // input_cpu.set_value<float>(1, i);
  }

  input_cu = input_cpu.clone();
  input_cu.to_cuda();

  std::shared_ptr<op::Layer> reduce_layer_cpu =
    std::make_shared<op::ReduceLayer>(
      base::DeviceType::kDeviceCPU, reduce_mode);

  std::shared_ptr<op::Layer> reduce_layer_cu =
    std::make_shared<op::ReduceLayer>(
      base::DeviceType::kDeviceCUDA, reduce_mode);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  reduce_layer_cu->set_cuda_config(config);

  reduce_layer_cpu->forward(input_cpu, output_cpu);
  reduce_layer_cu->forward(input_cu, output_cu);

  output_cu.to_cpu();

  constexpr float tol = 5e-1f;
  for (int i = 0; i < after_reduce_num; ++i) {
    const float cpu_v = output_cpu.at<float>(i);
    const float gpu_v = output_cu.at<float>(i);
    if (after_reduce_num < 10) {
      printf("cpu_v: %f, gpu_v: %f\n", cpu_v, gpu_v);
    }

    // 按你原逻辑：比较 abs(cpu) vs abs(gpu)
    float abs_tol = 1.0f;      // 给一点绝对兜底
    float rel_tol = 1e-5f;     // 先从 1e-5 试；不够再放到 3e-5 / 1e-4
    float diff = std::fabs(cpu_v - gpu_v);
    float bound = abs_tol + rel_tol * std::fabs(cpu_v);
    EXPECT_LE(diff, bound) << "cpu=" << cpu_v << " gpu=" << gpu_v;
  }
}

TEST(test_cu_reduce, test_0) {
  #if 1
  reduce(1024, 1024, 0);
  reduce(1024, 1024, 0);
  reduce(1024, 1024, 0);
  reduce(1024, 1024, 1);
  reduce(1024, 1024, 1);
  reduce(1024, 1024, 1);
  reduce(1024, 1024, 2);
  reduce(1024, 1024, 2);
  reduce(1024, 1024, 2);
  reduce(1024, 1024, 0);
  #endif
}

// TEST(test_cu_reduce, test_0) {
//   #if 1
//   reduce(128, 1024);
//   reduce(512, 1024);
//   reduce(1024, 1024);
//   reduce(2048, 7168);
//   #endif
// }

// TEST(test_cu_reduce, test_1) {
//   #if 1
//   reduce(129, 1024);
//   reduce(512, 1025);
//   reduce(1000, 1050);
//   reduce(2050, 7181);
//   #endif
// }
#endif
