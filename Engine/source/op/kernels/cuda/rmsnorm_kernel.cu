#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"

namespace kernel {
template <int32_t BLOCK_DIM>
__global__ void rmsnorm_kernel(float* in, float* weight, float* out,
                                int total_num, float eps) {
  const int tid = threadIdx.x;
  constexpr int vec_size = 4;
  const int vec_num = total_num / vec_size;
  const int vec_off = vec_size * vec_num;

// RMSNorm 公式
//  scale = Sigma(x^2) / d
//  rsqrt = 1 / (scale + eps) ^ 0.5
//  y = x * rsqrt * w
  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < vec_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  // 尾数处理
  for (int i = vec_off + tid; i < total_num; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  // 仅分配一个block, 使用shared_mem
  __shared__ float block_sum;

  // 仅分配一个block, 对该block内的sum进行规约,
  //  求出sum_block, 因为只有一个block, 这个block_sum就是总和
  //  3080 Shared_mem + L1: 128kb -> 128 * 1024 / 4 =  32768个fp32
  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    block_sum = sum;
  }
  __syncthreads();

  sum = block_sum; // 不使用block_sum, 因为sum在寄存器里, 更快
  float scale = rsqrtf(sum / static_cast<float>(total_num) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(weight);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < vec_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 weight_float4 = *(wei_pack + i);
    out_pack[i] = 
      make_float4(scale * in_float4.x * weight_float4.x,
                  scale * in_float4.y * weight_float4.y,
                  scale * in_float4.z * weight_float4.z,
                  scale * in_float4.w * weight_float4.w);
  }

  for (int i = vec_off + tid; i < total_num; i += blockDim.x) {
    out[i] = weight[i] * in[i] * scale;
  }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input,
                       const tensor::Tensor& weight,
                       tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#ifdef QWEN2_SUPPORT
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif
  const int32_t size = static_cast<int32_t>(input.size());
  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());

  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rmsnorm_kernel<128><<<1, threads_num, 0, stream_>>>(
      in_ptr, wei_ptr, out_ptr, size, eps);
  } else {
    rmsnorm_kernel<128><<<1, threads_num>>>(
      in_ptr, wei_ptr, out_ptr, size, eps);
  }
}
} // namespace kernel