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
// 切分target, 此分支, 每个block处理n行
//  将待处理的target行放入smem, source则根据index的位置, 选择性的加载到smem
//  smem: [target(target_cols), index(index_nums), source(target_cols)]
  if (para->block_per_target_row > 1) {
    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = blockIdx.x;

    extern __shared__ char smem[];

    uint32_t target_smem_size =
      math_cu::AlignUp<uint32_t>(
        para->ele_num_per_block * sizeof(float), SMEM_ALIGN);
    uint32_t index_smem_size =
      math_cu::AlignUp<uint32_t>(
        para->index_nums * sizeof(int32_t), SMEM_ALIGN);

    float* target_smem =
      reinterpret_cast<float*>(smem);
    int32_t* index_smem =
      reinterpret_cast<int32_t*>(smem + target_smem_size);

    for (int b = block_id; b < gridDim.x; b += gridDim.x) {
      // target: gmem -> smem
      for (int t = thread_id; t < para->ele_num_per_block; t += blockDim.x) {
        uint32_t target_offset = b * para->ele_num_per_block + t;
        target_smem[t] = input[target_offset];
      }
      for (int t = thread_id; t < para->index_nums; t += blockDim.x) {
        uint32_t index_offset = t;
        index_smem[t] = index[index_offset] < 0 ?
                        index[index_offset] + para->target_rows :
                        index[index_offset];
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
    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = blockIdx.x;

    extern __shared__ char smem[];

    uint32_t target_smem_size =
      math_cu::AlignUp<uint32_t>(
        para->target_cols * sizeof(float), SMEM_ALIGN);
    uint32_t index_smem_size =
      math_cu::AlignUp<uint32_t>(
        para->index_nums * sizeof(int32_t), SMEM_ALIGN);

    float* target_smem =
      reinterpret_cast<float*>(smem);
    int32_t* index_smem =
      reinterpret_cast<int32_t*>(smem + target_smem_size);

    for (int b = block_id; b < para->target_rows; b += gridDim.x) {
      // target: gmem -> smem
      for (int t = thread_id; t < para->target_cols; t += blockDim.x) {
        uint32_t target_offset = b * para->target_cols + t;
        target_smem[t] = input[target_offset];
      }
      for (int t = thread_id; t < para->index_nums; t += blockDim.x) {
        uint32_t index_offset = t;
        index_smem[t] = index[index_offset] < 0 ?
                        index[index_offset] + para->target_rows :
                        index[index_offset];
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

__global__ void index_add_kernel_GCU400_v0(const float* input,
                                           const int32_t* index,
                                           const float* source,
                                           float* output,
                                           para::index_add_para* para) {
// 切分index, 每个block处理n个index, 加载source的一行到smem, target始终在gmem
  if (para->block_per_source_row > 1) {
    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = blockIdx.x;

    __shared__ int32_t index_smem;

    // 只有source使用shared mem, 所以直接给float类型, 不使用char类型进行转换
    extern __shared__ float source_smem[];
    for (int b = block_id; b < gridDim.x; b += gridDim.x) {
      uint32_t rows_used_in_this_block = b / para->block_per_source_row;
      index_smem = index[rows_used_in_this_block] < 0 ?
                 index[rows_used_in_this_block] + para->target_rows :
                 index[rows_used_in_this_block];
      __syncthreads();

      for (int t = thread_id; t < para->ele_num_per_block; t += blockDim.x) {
        uint32_t source_offset = b * para->ele_num_per_block + t;
        source_smem[t] = source[source_offset];
      }
      __syncthreads();

      for (int t = thread_id; t < para->ele_num_per_block; t += blockDim.x) {
        uint32_t output_offset =
          index_smem * para->target_cols +
          (b % para->block_per_source_row) *
          para->ele_num_per_block + t;
        atomicAdd(&output[output_offset], source_smem[t]);
      }
      __syncthreads();
    }
  } else if (para->block_per_source_row == 1) {
    uint32_t thread_id = threadIdx.x;
    uint32_t block_id = blockIdx.x;
    __shared__ int32_t index_smem;

    // 只有source使用shared mem, 所以直接给float类型, 不使用char类型进行转换
    extern __shared__ float source_smem[];
    for (int b = block_id; b < para->index_nums; b += gridDim.x) {
      index_smem = index[b] < 0 ?
                 index[b] + para->target_rows :
                 index[b];
      __syncthreads();

      for (int t = thread_id; t < para->source_cols; t += blockDim.x) {
        uint32_t source_offset = b * para->source_cols + t;
        source_smem[t] = source[source_offset];
      }
      __syncthreads();

      for (int t = thread_id; t < para->source_cols; t += blockDim.x) {
        uint32_t output_offset = index_smem * para->target_cols + t;
        atomicAdd(&output[output_offset], source_smem[t]);
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
      //  将待处理的target行放入smem, source则根据index的位置, 选择性的加载到smem
      //  smem: [target(target_cols), index(index_nums), source(target_cols)]
      // 此种切分方式, add顺序与CPU保持一致, 结果不应该有很大误差, 接近比特一致
      // 问题: 如果index的值分布比较极端且rows较大(没有超发系数)
      //  eg: target: [512, 7168], index: [0, 0, 0, 0, 0, 0, 0, ..., 0]
      //  几乎没有并行性, 只会是thread0串行的进行数据操作
      blockNum = 256;
      threadNum = 512;

      para.block_per_target_row = 1;
      para.ele_num_per_block = para.target_cols;

      uint32_t target_smem_size =
        math_cu::AlignUp<uint32_t>(para.target_cols * para.bpe, SMEM_ALIGN);
      uint32_t index_smem_size =
        math_cu::AlignUp<uint32_t>(para.index_nums * para.bpe, SMEM_ALIGN);

      smem_size = target_smem_size + index_smem_size;
    } else {
      threadNum = 128;
      para.block_per_target_row =
        math_cu::CeilDiv<int32_t>(para.target_cols, threadNum);
      para.ele_num_per_block =
        math_cu::CeilDiv<int32_t>(para.target_cols, para.block_per_target_row);

      blockNum = para.target_rows * para.block_per_target_row;

      uint32_t target_smem_size =
        math_cu::AlignUp<uint32_t>(
          para.ele_num_per_block * para.bpe, SMEM_ALIGN);
      uint32_t index_smem_size =
        math_cu::AlignUp<uint32_t>(para.index_nums * para.bpe, SMEM_ALIGN);

      smem_size = target_smem_size + index_smem_size;
    }

    dim3 grid(blockNum);
    dim3 block(threadNum);

    para::index_add_para* para_d;
    cudaMalloc(&para_d, sizeof(para::index_add_para));
    cudaMemcpy(para_d, &para,
               sizeof(para::index_add_para), cudaMemcpyHostToDevice);

#if DEBUG
    printf("Launching GCU300 <<<%u, %u, %u>>>\n",
      blockNum, threadNum, smem_size);
#endif
    // GCU300的切分方式可能会使smem_size大于device的默认smem大小(40KB)
    //  因此, 强行分配
    // GCU400的切分方式则没有可能
    if (smem_size > 48 * 1024) {
      cudaFuncSetAttribute(
        index_add_kernel_GCU300_v0,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size // e.g. 61440 or up to 102400 on CC8.6
      );
    }
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Launch GCU300 kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(para_d);
  } else if (para.enflame_device == para::EnflameDevice::GCU400) {
    if (para.index_nums >= 256) {
      // 切分index, 每个block至少处理一个index(一行source),
      //  512个thread处理一行中的所有元素
      // 这种切分方式会使index并行
      //  eg: index[0, 0, 1, 1, 2], thread2和thread3同时要对target的第1行进行操作,
      //  虽然可以通过atomicAdd来解决多线程之间的竞争问题, 但是无法保证累加顺序,
      //  无法做到和GCU300一样的计算精度(与cpu结果接近比特一致)
      blockNum = 256;
      threadNum = 512;
      dim3 grid(blockNum);
      dim3 block(threadNum);

      para.block_per_source_row = 1;

      uint32_t index_smem_size = 1 * para.bpe;
      uint32_t source_smem_size =
        math_cu::AlignUp<uint32_t>(para.target_cols * para.bpe, SMEM_ALIGN);
      smem_size = source_smem_size + index_smem_size;
    } else {
      threadNum = 128;
      para.block_per_source_row =
        math_cu::CeilDiv<int32_t>(para.source_cols, threadNum);
      para.ele_num_per_block =
        math_cu::CeilDiv<int32_t>(para.source_cols, para.block_per_source_row);

      blockNum = para.index_nums * para.block_per_source_row;

        uint32_t index_smem_size = 1 * para.bpe;
      uint32_t source_smem_size =
        math_cu::AlignUp<uint32_t>(
          para.ele_num_per_block * para.bpe, SMEM_ALIGN);

      smem_size = source_smem_size + index_smem_size;
    }

    dim3 grid(blockNum);
    dim3 block(threadNum);

    para::index_add_para* para_d;
    cudaMalloc(&para_d, sizeof(para::index_add_para));
    cudaMemcpy(para_d, &para,
               sizeof(para::index_add_para), cudaMemcpyHostToDevice);

#if DEBUG
    printf("Launching GCU400 <<<%u, %u, %u>>>\n",
      blockNum, threadNum, smem_size);
#endif
    if (stream) {
      cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
      index_add_kernel_GCU400_v0<<<grid, block, smem_size, stream_>>>(
        target.ptr<float>(), index.ptr<int32_t>(),
        source.ptr<float>(), output.ptr<float>(), para_d);
    } else {
      index_add_kernel_GCU400_v0<<<grid, block, smem_size>>>(
        target.ptr<float>(), index.ptr<int32_t>(),
        source.ptr<float>(), output.ptr<float>(), para_d);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Launch GCU400 kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(para_d);
  } else {
    printf(" ================== ERROR: invaild enflame_device\n");
  }
}
} // namespace kernel
