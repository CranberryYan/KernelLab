#include <cuda_runtime_api.h>
#include "math_utils.cuh"
#include "base/para.h"
#include "index_add_kernel.cuh"

namespace kernel {
#define DEBUG 1
__global__ void index_add_kernel_GCU300_v0(const float* input,
                                           const int32_t* index,
                                           const float* source,
                                           float* output,
                                           para::index_add_para* para) {
  if (para->block_per_target_row > 1) {
    // 切分target, 此分支, 每个block处理n行
    //  将待处理的target行放入L2, source则根据index的位置, 选择性的加载到L2
    //  L2: [target(target_cols), index(index_nums), source(target_cols)]
    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = blockIdx.x;

    extern __shared__ char smem[];

    uint32_t target_smem_size =
      math_cu::AlignUp<uint32_t>(para->ele_num_per_block * sizeof(float), 128);
    uint32_t index_smem_size =
      math_cu::AlignUp<uint32_t>(para->index_nums * sizeof(int32_t), 128);

    float* target_smem =
      reinterpret_cast<float*>(smem);
    int32_t* index_smem =
      reinterpret_cast<int32_t*>(smem + target_smem_size);

    // 数据流
    for (int b = block_id; b < gridDim.x; b += gridDim.x) {
      // target: L3 -> L2
      for (int t = thread_id; t < para->ele_num_per_block; t += blockDim.x) {
        uint32_t target_offset = b * para->ele_num_per_block + t;
        target_smem[t] = input[target_offset];
      }
      for (int t = thread_id; t < para->index_nums; t += blockDim.x) {
        uint32_t index_offset = t;
        index_smem[t] = index[index_offset];
      }
      __syncthreads();

      for (int i = 0; i < para->index_nums; ++i) {
        int32_t target_rows_used = index_smem[i];
        uint32_t row_id = b / para->block_per_target_row;
        if (row_id == target_rows_used) {
          for (int t = thread_id;
               t < para->ele_num_per_block; t += blockDim.x) {
            target_smem[t] += source[i * para->source_cols + t];
          }
        }
      }
      __syncthreads();

      for (int t = thread_id; t < para->ele_num_per_block; t += blockDim.x) {
        uint32_t target_offset = b * para->ele_num_per_block + t;
        output[target_offset] = target_smem[t];
      }
      __syncthreads();
    }
  } else if (para->block_per_target_row == 1) {
    // 切分target, 此分支, 每个block处理n行
    //  将待处理的target行放入L2, source则根据index的位置, 选择性的加载到L2
    //  L2: [target(target_cols), index(index_nums), source(target_cols)]
    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = blockIdx.x;

    extern __shared__ char smem[];

    uint32_t target_smem_size =
      math_cu::AlignUp<uint32_t>(para->target_cols * sizeof(float), 128);
    uint32_t index_smem_size =
      math_cu::AlignUp<uint32_t>(para->index_nums * sizeof(int32_t), 128);

    float* target_smem =
      reinterpret_cast<float*>(smem);
    int32_t* index_smem =
      reinterpret_cast<int32_t*>(smem + target_smem_size);

    // 数据流
    for (int b = block_id; b < para->target_rows; b += gridDim.x) {
      // target: L3 -> L2
      for (int t = thread_id; t < para->target_cols; t += blockDim.x) {
        uint32_t target_offset = b * para->target_cols + t;
        target_smem[t] = input[target_offset];
      }
      for (int t = thread_id; t < para->index_nums; t += blockDim.x) {
        uint32_t index_offset = t;
        index_smem[t] = index[index_offset];
      }
      __syncthreads();

      for (int i = 0; i < para->index_nums; ++i) {
        int32_t target_rows_used = index_smem[i];
        if (b == target_rows_used) {
          for (int t = thread_id; t < para->target_cols; t += blockDim.x) {
            target_smem[t] += source[i * para->target_cols + t];
          }
        }
      }
      __syncthreads();

      for (int t = thread_id; t < para->target_cols; t += blockDim.x) {
        uint32_t target_offset = b * para->target_cols + t;
        output[target_offset] = target_smem[t];
      }
      __syncthreads();
    }
  }
}

void index_add_kernel_cu(const tensor::Tensor& target,
                         const tensor::Tensor& index,
                         const tensor::Tensor& source,
                         tensor::Tensor& output,
                         para::index_add_para para,
                         void* stream) {
#if DEBUG
  std::cout
    << "target_dims: [" 
    << para.target_dims[0]<< ", " << para.target_dims[1] << "]\n"
    << "index_dims:  [" << para.index_dims[0]  << "]\n"
    << "source_dims: [" 
    << para.source_dims[0] << ", " << para.source_dims[1] << "]\n"
    << "target_rows: " << para.target_rows  << "\n"
    << "target_cols: " << para.target_cols  << "\n"
    << "index_nums:  " << para.index_nums   << "\n"
    << "source_rows: " << para.source_rows  << "\n"
    << "source_cols: " << para.source_cols  << "\n";
#endif

  uint32_t smem_size = 0;
  uint32_t blockNum = 0;
  uint32_t threadNum = 0;
  if (para.enflame_device == para::EnflameDevice::GCU300) {
    if (para.target_rows >= 256) {
      // 切分target, 此分支, 每个block处理n行
      //  将待处理的target行放入L2, source则根据index的位置, 选择性的加载到L2
      //  L2: [target(target_cols), index(index_nums), source(target_cols)]
      // 此种切分方式, add顺序与CPU保持一致, 结果不应该有很大误差, 接近比特一致
      blockNum = 256;
      threadNum = 512;

      para.block_per_target_row = 1;
      para.ele_num_per_block = para.target_cols;

      uint32_t target_smem_size =
        math_cu::AlignUp<uint32_t>(para.target_cols * para.bpe, 128);
      uint32_t index_smem_size =
        math_cu::AlignUp<uint32_t>(para.index_nums * para.bpe, 128);

      smem_size = target_smem_size + index_smem_size;
    } else {
      para.block_per_target_row =
        math_cu::CeilDiv<int32_t>(para.target_cols, 128);
      para.ele_num_per_block =
        math_cu::CeilDiv(para.target_cols, para.block_per_target_row);

      threadNum = 128;
      blockNum = para.target_rows * para.block_per_target_row;

      uint32_t target_smem_size =
        math_cu::AlignUp<uint32_t>(para.ele_num_per_block * para.bpe, 128);
      uint32_t index_smem_size =
        math_cu::AlignUp<uint32_t>(para.index_nums * para.bpe, 128);

      smem_size = target_smem_size + index_smem_size;
    }

    dim3 grid(blockNum);
    dim3 block(threadNum);

    para::index_add_para* para_d;
    cudaMalloc(&para_d, sizeof(para::index_add_para));
    cudaMemcpy(para_d, &para,
               sizeof(para::index_add_para), cudaMemcpyHostToDevice);

    if (stream) {
      cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
      index_add_kernel_GCU300_v0<<<grid, block, smem_size, stream_>>>(
        target.ptr<float>(), index.ptr<int32_t>(),
        source.ptr<float>(), output.ptr<float>(), para_d);
    } else {
      index_add_kernel_GCU300_v0<<<grid, block, smem_size>>>(
        target.ptr<float>(), index.ptr<int32_t>(),
        source.ptr<float>(), output.ptr<float>(), para_d);
    }
  } else if (para.enflame_device == para::EnflameDevice::GCU400) {
    // 切分index, 每个block至少处理一个index, 512个thread处理一行中的所有元素
    uint32_t blockNum = 256;
    uint32_t threadNum = 512;
    dim3 grid(blockNum);
    dim3 block(threadNum);
  } else {
    printf(" ================== ERROR: invaild enflame_device\n");
  }
}
} // namespace kernel
