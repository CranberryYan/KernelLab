#include "swiglu_kernel.cuh"

namespace kernel
{
__global__ void swiglu_kernel(int size, const float* input_1,
                              const float* input_2, float* output) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= size) {
    return;
  }

  for (int i = gid; i < size; i += blockDim.x) {
    float sigmoid = 1.0f / (1.0f + std::exp(-input_1[i]));
    output[i] = (input_1[i] * sigmoid) * input_2[i];
  }
}

void swiglu_kernel_cu(const tensor::Tensor& input1,
                      const tensor::Tensor& input2,
                      tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  int threads = 256;
  int blocks = 128;

  if (!stream) {
    swiglu_kernel<<<blocks, threads>>>(size,
                                       input1.ptr<float>(),
                                       input2.ptr<float>(),
                                       output.ptr<float>());
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel<<<blocks, threads, 0, stream_>>>(size,
                                                   input1.ptr<float>(), 
                                                   input2.ptr<float>(),
                                                   output.ptr<float>());
  }
}
} // namespace kernel
