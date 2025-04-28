#include <cuda_runtime_api.h>
#include "math_utils.cuh"
#include "base/para.h"
#include "scatter_kernel.cuh"

namespace kernel {
__global__ void scatter_kernel_v0(const float* input, const int32_t* index,
                                  const float* src, float* output,
                                  para::scatter_para* para) {
  uint32_t block_id = blockIdx.x;
  uint32_t thread_id = threadIdx.x;
  uint32_t input_ele_num_per_block = para->input_ele_num_per_block;
  uint32_t src_ele_num_per_block = para->src_ele_num_per_block;
  uint32_t index_ele_num_per_block = para->index_ele_num_per_block;
  uint32_t index_offset = block_id * index_ele_num_per_block + thread_id;
  uint32_t src_offset = block_id * src_ele_num_per_block + thread_id;
  if (index_offset >= para->index_ele_num) {
    return;
  }

  // 数据流
  uint32_t shared_src_mem_size =
    math_cu::AlignUp<uint32_t>(
      src_ele_num_per_block * sizeof(float), 128);
  uint32_t shared_index_mem_size =
    math_cu::AlignUp<uint32_t>(
      index_ele_num_per_block * sizeof(uint32_t), 128);

  extern __shared__ char smem[];
  int* shared_index = reinterpret_cast<int*>(smem);
  float* shared_src = reinterpret_cast<float*>(smem + shared_index_mem_size);

  shared_index[thread_id] = index[index_offset] < 0 ?
                            index[index_offset] + input_ele_num_per_block :
                            index[index_offset];
  shared_src[thread_id] = src[src_offset];
  __syncthreads();

  // 计算流
  // 对于Update来说, 保证严格顺序, 串行
  // 对于Add来说, 不是严格顺序, 但是要使用原子加法防止多线程之间的竞争, 保证最后结果正确即可
  if (para->op_type == para::ScatterOpType::Update) {
    if (thread_id == 0) {
      for (int i = 0; i < index_ele_num_per_block; ++i) {
        uint32_t output_offset =
          block_id * input_ele_num_per_block + shared_index[i];
        output[output_offset] = shared_src[i];
      }
    }
  } else if (para->op_type == para::ScatterOpType::Add) {
    uint32_t output_offset =
      block_id * input_ele_num_per_block + shared_index[thread_id];
    atomicAdd(&output[output_offset], shared_src[thread_id]);
  }
  // if (thread_id == 0) {
  //   if (para->op_type == para::ScatterOpType::Add) {
  //     for (int i = 0; i < index_ele_num_per_block; ++i) {
  //       uint32_t output_offset =
  //         block_id * input_ele_num_per_block + shared_index[i];
  //       output[output_offset] += shared_src[i];
  //     }
  //   } else if (para->op_type == para::ScatterOpType::Update) {
  //     for (int i = 0; i < index_ele_num_per_block; ++i) {
  //       uint32_t output_offset =
  //         block_id * input_ele_num_per_block + shared_index[i];
  //       output[output_offset] = shared_src[i];
  //     }
  //   }
  // }

  // 问题: 这里应该是原子操作???, 否则这里的线程之间的竞争, 会导致output不一致???
  //  原子操作还不够, 这里最重要的是add, update要按照cols(thread_id)的顺序来进行
  // if (para->op_type == para::ScatterOpType::Add) {
  //   atomicAdd(&output[target_offset], src[block_id * blockDim.x + thread_id]);
  // } else if (para->op_type == para::ScatterOpType::Update) {
  //   atomicExch(&output[target_offset], src[block_id * blockDim.x + thread_id]);
  // }
}

void scatter_kernel_cu(const tensor::Tensor &input,
                       const tensor::Tensor &src,
                       const tensor::Tensor &index,
                       tensor::Tensor &output,
                       para::scatter_para para,
                       void* stream) {
  uint32_t thread_num = para.thread_num;
  uint32_t block_num = para.block_num;

  dim3 grid(block_num);
  dim3 block(thread_num);

  uint32_t shared_src_mem_size =
    math_cu::AlignUp<uint32_t>(
      para.src_ele_num_per_block * sizeof(float), 128);
  uint32_t shared_index_mem_size =
    math_cu::AlignUp<uint32_t>(
      para.index_ele_num_per_block * sizeof(uint32_t), 128);
  uint32_t shared_mem_size = shared_src_mem_size + shared_index_mem_size;

  para::scatter_para* para_device;
  cudaMalloc(&para_device, sizeof(para::scatter_para));
  cudaMemcpy(para_device, &para,
    sizeof(para::scatter_para), cudaMemcpyHostToDevice);

  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    scatter_kernel_v0<<<grid, block, shared_mem_size, stream_>>>(
      input.ptr<float>(), index.ptr<int32_t>(),
      src.ptr<float>(), output.ptr<float>(), para_device);
  } else {
    scatter_kernel_v0<<<grid, block, shared_mem_size>>>(
      input.ptr<float>(), index.ptr<int32_t>(),
      src.ptr<float>(), output.ptr<float>(), para_device);
  }

  cudaFree(para_device);
}
} // namespace kernel
