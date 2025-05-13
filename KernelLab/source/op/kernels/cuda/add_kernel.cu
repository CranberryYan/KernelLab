#include "add_kernel.cuh"
#include <cuda_runtime_api.h>

// add_kernel_v0
//  baseline
//  Memory Throughput [Gbyte/second]	516.68
// add_kernel_v1
//  向量化
//  
namespace kernel {
struct Vec {
  using Type = float4;
  static constexpr uint32_t ele_num = 4;
};

__global__ void add_kernel_v0(int32_t total_num,
                              const float* input1, const float* input2,
                              float* output) {
  int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total_num) { return; }
  output[gid] = input1[gid] + input2[gid];
}

__global__ void add_kernel_v1(int32_t total_num,
                              const float* input1, const float* input2,
                              float* output) {
  int vec_num = Vec::ele_num;
  int package_num = total_num / vec_num;
  using Vec_t = typename Vec::Type;
  int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= package_num) { return; }

  const Vec_t* in1_vec = reinterpret_cast<const Vec_t*>(input1);
  const Vec_t* in2_vec = reinterpret_cast<const Vec_t*>(input2);
  Vec_t* out_vec = reinterpret_cast<Vec_t*>(output);
  out_vec[gid].x = in1_vec[gid].x + in2_vec[gid].x;
  out_vec[gid].y = in1_vec[gid].y + in2_vec[gid].y;
  out_vec[gid].z = in1_vec[gid].z + in2_vec[gid].z;
  out_vec[gid].w = in1_vec[gid].w + in2_vec[gid].w;
}

void add_kernel_cu(const tensor::Tensor& input1,
                   const tensor::Tensor& input2,
                   tensor::Tensor& output,
                   para::add_para para,
                   void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  CHECK_EQ(input1.size(), input2.size());
  CHECK_EQ(input1.size(), output.size());

  int32_t ele_num = para.ele_num;
  int32_t thread_num = para.thread_num;
  int32_t block_num = para.block_num;

  dim3 grid(block_num);
  dim3 block(thread_num);
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_v0<<<grid, block, 0, stream_>>>(
      ele_num, input1.ptr<float>(), input2.ptr<float>(), output.ptr<float>());
  } else {
    add_kernel_v0<<<grid, block>>>(
      ele_num, input1.ptr<float>(), input2.ptr<float>(), output.ptr<float>());
  }
}
} // namespace kernel
