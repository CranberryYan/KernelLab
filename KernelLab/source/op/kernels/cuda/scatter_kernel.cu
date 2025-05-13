#include <cuda_runtime_api.h>
#include "math_utils.cuh"
#include "base/para.h"
#include "scatter_kernel.cuh"

namespace kernel {
//TODO: thread的分配, 大shape会不够用
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
      src_ele_num_per_block * sizeof(float), SMEM_ALIGN);
  uint32_t shared_index_mem_size =
    math_cu::AlignUp<uint32_t>(
      index_ele_num_per_block * sizeof(uint32_t), SMEM_ALIGN);

  extern __shared__ char smem[];
  int* shared_index = reinterpret_cast<int*>(smem);
  float* shared_src = reinterpret_cast<float*>(smem + shared_index_mem_size);

  shared_index[thread_id] = index[index_offset] < 0 ?
                            index[index_offset] + input_ele_num_per_block :
                            index[index_offset];
  shared_src[thread_id] = src[src_offset];
  __syncthreads();

  // 计算流
  // Update: 保证严格顺序, 串行
  // Add: 不是严格顺序, 但是要使用原子加法防止多线程之间的竞争, 保证最后结果正确即可
  if (para->op_type == para::ScatterOpType::Scatter_Update) {
    if (thread_id == 0) {
      for (int i = 0; i < index_ele_num_per_block; ++i) {
        uint32_t output_offset =
          block_id * input_ele_num_per_block + shared_index[i];
        output[output_offset] = shared_src[i];
      }
    }
  } else if (para->op_type == para::ScatterOpType::Scatter_Add) {
    uint32_t output_offset =
      block_id * input_ele_num_per_block + shared_index[thread_id];
    atomicAdd(&output[output_offset], shared_src[thread_id]);
  }
  // if (thread_id == 0) {
  //   if (para->op_type == para::ScatterOpType::Scatter_Add) {
  //     for (int i = 0; i < index_ele_num_per_block; ++i) {
  //       uint32_t output_offset =
  //         block_id * input_ele_num_per_block + shared_index[i];
  //       output[output_offset] += shared_src[i];
  //     }
  //   } else if (para->op_type == para::ScatterOpType::Scatter_Update) {
  //     for (int i = 0; i < index_ele_num_per_block; ++i) {
  //       uint32_t output_offset =
  //         block_id * input_ele_num_per_block + shared_index[i];
  //       output[output_offset] = shared_src[i];
  //     }
  //   }
  // }

  // 问题: 这里应该是原子操作???, 否则这里的线程之间的竞争, 会导致output不一致???
  //  原子操作还不够, 这里最重要的是add, update要按照cols(thread_id)的顺序来进行
  // if (para->op_type == para::ScatterOpType::Scatter_Add) {
  //   atomicAdd(&output[target_offset], src[block_id * blockDim.x + thread_id]);
  // } else if (para->op_type == para::ScatterOpType::Scatter_Update) {
  //   atomicExch(&output[target_offset], src[block_id * blockDim.x + thread_id]);
  // }
}

__global__ void gather_kernel_v0(const float* input,
                                 const int32_t* index,
                                 float* output,
                                 para::scatter_para* para) {
  uint32_t block_id = blockIdx.x;
  uint32_t thread_id = threadIdx.x;

  extern __shared__ char smem[];
  uint32_t index_smem_size =
    math_cu::AlignUp<uint32_t>(
      para->index_ele_num_per_block * sizeof(int32_t), SMEM_ALIGN);
  int32_t* index_smem = reinterpret_cast<int32_t*>(smem);
  float* input_smem = reinterpret_cast<float*>(smem + index_smem_size);

  for (int b = block_id; b < para->input_rows; b += gridDim.x) {
    // TODO: 向量化的读取
    // index: gmem -> smem
    for (int t = thread_id;
         t < para->index_ele_num_per_block; t += blockDim.x) {
      int32_t index_offset = b * para->index_ele_num_per_block + t;
      int32_t raw_index = index[index_offset];
      index_smem[t] = raw_index < 0 ?
                      raw_index + para->input_cols :
                      raw_index;
    }
    __syncthreads();

    // input -> output
    // 1. gmem -> gmem(从index中挑选)
    // 2. gmem -> smem(一整行) -> gmem(从index中挑选)
    //  (不一定是最优解, 但是为了练习, 选择该种切分方式)
    // input: gmem -> smem
    for (int t = thread_id;
         t < para->input_ele_num_per_block; t += blockDim.x) {
      int32_t input_offset = b * para->input_ele_num_per_block + t;
      input_smem[t] = input[input_offset];
    }
    __syncthreads();

    // input(smem) -> output(gmem)
    for (int t = thread_id;
         t < para->index_ele_num_per_block; t += blockDim.x) {
      int32_t output_offset = b * para->index_ele_num_per_block + t;
      int32_t col_used = index_smem[t];
      output[output_offset] = input_smem[col_used];
    }
    __syncthreads();
  }
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

  if (para.op_type == para::ScatterOpType::Scatter_Add ||
      para.op_type == para::ScatterOpType::Scatter_Update) {
    uint32_t shared_src_mem_size =
      math_cu::AlignUp<uint32_t>(
        para.src_ele_num_per_block * sizeof(float), SMEM_ALIGN);
    uint32_t shared_index_mem_size =
      math_cu::AlignUp<uint32_t>(
        para.index_ele_num_per_block * sizeof(uint32_t), SMEM_ALIGN);
    uint32_t smem_size = shared_src_mem_size + shared_index_mem_size;
    if (smem_size > 48 * 1024) {
      cudaFuncSetAttribute(
        scatter_kernel_v0,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size // e.g. 61440 or up to 102400 on CC8.6
      );
    }

    para::scatter_para* para_d;
    cudaMalloc(&para_d, sizeof(para::scatter_para));
    cudaMemcpy(para_d, &para,
      sizeof(para::scatter_para), cudaMemcpyHostToDevice);

    if (stream) {
      cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
      scatter_kernel_v0<<<grid, block, smem_size, stream_>>>(
        input.ptr<float>(), index.ptr<int32_t>(),
        src.ptr<float>(), output.ptr<float>(), para_d);
    } else {
      scatter_kernel_v0<<<grid, block, smem_size>>>(
        input.ptr<float>(), index.ptr<int32_t>(),
        src.ptr<float>(), output.ptr<float>(), para_d);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Launch scatter_kernel_v0 kernel error: %s\n",
        cudaGetErrorString(err));
    }

    cudaFree(para_d);
  } else if (para.op_type == para::ScatterOpType::Gather) {
    uint32_t input_smem_size =
      math_cu::AlignUp<uint32_t>(
        para.input_ele_num_per_block * sizeof(float), SMEM_ALIGN);
    uint32_t index_smem_size =
      math_cu::AlignUp<uint32_t>(
        para.index_ele_num_per_block * sizeof(int32_t), SMEM_ALIGN);
    uint32_t smem_size = index_smem_size + input_smem_size;

    para::scatter_para* para_d;
    cudaMalloc(&para_d, sizeof(para::scatter_para));
    cudaMemcpy(para_d, &para,
      sizeof(para::scatter_para), cudaMemcpyHostToDevice);

    if (smem_size > 48 * 1024) {
      cudaFuncSetAttribute(
        gather_kernel_v0,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size // e.g. 61440 or up to 102400 on CC8.6
      );
    }

    if (stream) {
      cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
      gather_kernel_v0<<<grid, block, smem_size, stream_>>>(
        input.ptr<float>(), index.ptr<int32_t>(), output.ptr<float>(), para_d);
    } else {
      gather_kernel_v0<<<grid, block, smem_size>>>(
        input.ptr<float>(), index.ptr<int32_t>(), output.ptr<float>(), para_d);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Launch gather_kernel_v0 kernel error: %s\n",
        cudaGetErrorString(err));
    }

    cudaFree(para_d);
  } else {
    printf(" ERROR: invaild op_type\n");
  }
}
} // namespace kernel
