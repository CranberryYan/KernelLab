#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include "../include/op/histogram.h"
#include "../source/op/kernels/kernels_interface.h"

#if 1
void histogram_uniform(unsigned int ele_num, int32_t low, int32_t high) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int32_t length = high - low;
  tensor::Tensor input_cpu(base::DataType::kDataTypeInt32, ele_num,
                           true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeInt32, length,
                           true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeInt32, ele_num,
                           true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeInt32, length,
                           true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist_float(low, high);
  for (int i = 0; i < ele_num; ++i) {
    float input_tmp = dist_float(mt);
    input_cpu.set_value<int32_t>(static_cast<int32_t>(input_tmp), i);
    // input_cpu.set_value<float>(1, i);
  }

  for (int i = 0; i < length; ++i) {
    output_cpu.set_value<int32_t>(0, i);
  }

  input_cu = input_cpu.clone();
  input_cu.to_cuda();

  output_cu = output_cpu.clone();
  output_cu.to_cuda();

  std::shared_ptr<op::Layer> histogram_layer_cpu =
    std::make_shared<op::HistogramLayer>(
      low, high, base::DeviceType::kDeviceCPU);

  std::shared_ptr<op::Layer> histogram_layer_cu =
    std::make_shared<op::HistogramLayer>(
      low, high, base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  histogram_layer_cu->set_cuda_config(config);

  histogram_layer_cpu->forward(input_cpu, output_cpu);
  histogram_layer_cu->forward(input_cu, output_cu);

  output_cu.to_cpu();

  for (int i = 0; i < length; ++i) {
    const int32_t cpu_v = output_cpu.at<int32_t>(i);
    const int32_t gpu_v = output_cu.at<int32_t>(i);
    if (i < 10) {
      // printf("cpu_v: %d, gpu_v: %d\n", cpu_v, gpu_v);
    }

    EXPECT_EQ(cpu_v, gpu_v) << "cpu=" << cpu_v << " gpu=" << gpu_v;
  }
}

void histogram_hotspot_80(unsigned int ele_num, int32_t low, int32_t high) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int32_t length = high - low;
  tensor::Tensor input_cpu(base::DataType::kDataTypeInt32, ele_num,
                           true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeInt32, length,
                            true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeInt32, ele_num,
                          true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeInt32, length,
                           true, alloc_cu, nullptr);

  std::random_device rd;
  std::mt19937 mt(rd());

  // 决定是否走热点分布：80% 走热点，20% 走均匀分布
  std::uniform_real_distribution<float> prob_dist(0.f, 1.f);

  // 20% 的尾部分布：仍然均匀落到整个 [low, high)
  std::uniform_int_distribution<int32_t> uniform_bin_dist(low, high - 1);

  // 热点二选一：如果 length >= 2，则在两个热点 bin 中随机选一个
  std::uniform_int_distribution<int32_t> hot_bin_pick(0, 1);

  const int32_t hot_bin0 = low;
  const int32_t hot_bin1 = (length >= 2) ? (low + 1) : low;

  for (uint32_t i = 0; i < ele_num; ++i) {
    int32_t val = 0;
    float p = prob_dist(mt);

    if (p < 0.8f) {
      // 80% 概率集中到 1~2 个热点 bin
      if (length >= 2) {
        val = (hot_bin_pick(mt) == 0) ? hot_bin0 : hot_bin1;
      } else {
        val = hot_bin0;
      }
    } else {
      // 20% 概率均匀分布
      val = uniform_bin_dist(mt);
    }

    input_cpu.set_value<int32_t>(val, i);
  }

  for (int i = 0; i < length; ++i) {
    output_cpu.set_value<int32_t>(0, i);
  }

  #if 1
  input_cu = input_cpu.clone();
  input_cu.to_cuda();

  output_cu = output_cpu.clone();
  output_cu.to_cuda();

  std::shared_ptr<op::Layer> histogram_layer_cpu =
    std::make_shared<op::HistogramLayer>(
      low, high, base::DeviceType::kDeviceCPU);

  std::shared_ptr<op::Layer> histogram_layer_cu =
    std::make_shared<op::HistogramLayer>(
      low, high, base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  histogram_layer_cu->set_cuda_config(config);
  #endif

  #if 1
  histogram_layer_cpu->forward(input_cpu, output_cpu);
  histogram_layer_cu->forward(input_cu, output_cu);

  output_cu.to_cpu();
  #endif

  #if 1
  for (int i = 0; i < length; ++i) {
    const int32_t cpu_v = output_cpu.at<int32_t>(i);
    const int32_t gpu_v = output_cu.at<int32_t>(i);

    if (i < 10) {
      // printf("bin %d: cpu=%d, gpu=%d\n", i, cpu_v, gpu_v);
    }

    EXPECT_EQ(cpu_v, gpu_v) << "cpu=" << cpu_v << " gpu=" << gpu_v;
  }
  #endif
}

void histogram_signal(unsigned int ele_num, int32_t low, int32_t high) {
  std::shared_ptr<base::CPUDeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::CUDADeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  int32_t length = high - low;
  tensor::Tensor input_cpu(base::DataType::kDataTypeInt32, ele_num,
                           true, alloc_cpu, nullptr);
  tensor::Tensor output_cpu(base::DataType::kDataTypeInt32, length,
                           true, alloc_cpu, nullptr);
  tensor::Tensor input_cu(base::DataType::kDataTypeInt32, ele_num,
                           true, alloc_cu, nullptr);
  tensor::Tensor output_cu(base::DataType::kDataTypeInt32, length,
                           true, alloc_cu, nullptr);

  for (int i = 0; i < ele_num; ++i) {
    input_cpu.set_value<int32_t>(10, i);
    // input_cpu.set_value<float>(1, i);
  }

  for (int i = 0; i < length; ++i) {
    output_cpu.set_value<int32_t>(0, i);
  }

  input_cu = input_cpu.clone();
  input_cu.to_cuda();

  output_cu = output_cpu.clone();
  output_cu.to_cuda();

  std::shared_ptr<op::Layer> histogram_layer_cpu =
    std::make_shared<op::HistogramLayer>(
      low, high, base::DeviceType::kDeviceCPU);

  std::shared_ptr<op::Layer> histogram_layer_cu =
    std::make_shared<op::HistogramLayer>(
      low, high, base::DeviceType::kDeviceCUDA);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<kernel::CudaConfig> config =
    std::make_shared<kernel::CudaConfig>();
  config->stream = stream;
  histogram_layer_cu->set_cuda_config(config);

  histogram_layer_cpu->forward(input_cpu, output_cpu);
  histogram_layer_cu->forward(input_cu, output_cu);

  output_cu.to_cpu();

  for (int i = 0; i < length; ++i) {
    const int32_t cpu_v = output_cpu.at<int32_t>(i);
    const int32_t gpu_v = output_cu.at<int32_t>(i);
    if (i < 10) {
      // printf("cpu_v: %d, gpu_v: %d\n", cpu_v, gpu_v);
    }

    EXPECT_EQ(cpu_v, gpu_v) << "cpu=" << cpu_v << " gpu=" << gpu_v;
  }
}

TEST(test_cu_histogram, test_uniform) {
  #if 1
  histogram_uniform(4194304, 0, 256);
  histogram_uniform(4194304, 0, 256);
  histogram_uniform(4194304, 0, 256);
  histogram_uniform(4194304, 0, 256);
  histogram_uniform(4194304, 0, 256);
  histogram_uniform(4194304, 0, 256);
  histogram_uniform(4194304, 0, 256);
  histogram_uniform(4194304, 0, 256);
  histogram_uniform(4194304, 0, 256);
  histogram_uniform(4194304, 0, 256);
  #endif
}

TEST(test_cu_histogram, test_hotspot_80) {
  #if 1
  histogram_hotspot_80(4194304, 0, 256);
  histogram_hotspot_80(4194304, 0, 256);
  histogram_hotspot_80(4194304, 0, 256);
  histogram_hotspot_80(4194304, 0, 256);
  histogram_hotspot_80(4194304, 0, 256);
  histogram_hotspot_80(4194304, 0, 256);
  histogram_hotspot_80(4194304, 0, 256);
  histogram_hotspot_80(4194304, 0, 256);
  histogram_hotspot_80(4194304, 0, 256);
  histogram_hotspot_80(4194304, 0, 256);
  #endif
}

TEST(test_cu_histogram, test_signal) {
  #if 1
  histogram_signal(4194304, 0, 256);
  histogram_signal(4194304, 0, 256);
  histogram_signal(4194304, 0, 256);
  histogram_signal(4194304, 0, 256);
  histogram_signal(4194304, 0, 256);
  histogram_signal(4194304, 0, 256);
  histogram_signal(4194304, 0, 256);
  histogram_signal(4194304, 0, 256);
  histogram_signal(4194304, 0, 256);
  histogram_signal(4194304, 0, 256);
  #endif
}
#endif
