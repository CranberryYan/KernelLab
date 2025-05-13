#include "embedding_kernel.cuh"

namespace kernel {

// input: [token_nums]
// output: [token_nums, hidden_units]
__global__ void embedding_kernel(int32_t vocab_size, int32_t token_num,
                            int32_t weight_dim,
                            const int32_t* input_ptr,
                            const float* weight_ptr,
                            float* output_ptr) {
  // 每个token由一个block负责, hidden_units为128的一维向量,
  //  每个thread负责其中一个数
  int token_id = blockIdx.x;
  if (token_id >= token_num) {
    return;
  }

  int32_t token = input_ptr[token_id];
  if (token >= vocab_size) {
    printf("ERROR: vocab_size is %d, token is %d, token > vocab_size\n",
      vocab_size, token);
    return;
  }

  int32_t output_ptr_offset = token_id * weight_dim;
  int32_t weight_ptr_offset = token * weight_dim;
  for (int i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr[output_ptr_offset + i] =
      weight_ptr[weight_ptr_offset + i];
  }
}

void embedding_kernel_cu(const tensor::Tensor& input,
                         const tensor::Tensor& weight,
                         tensor::Tensor& output,
                         int32_t vocab_size, void* stream) {
  // 模型的第一层, CPU2CUDA
  tensor::Tensor input_cu;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda();
  } else if (input.device_type() == base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
  }

  const int32_t input_num = static_cast<int32_t>(input.size()); // token总数
  const int32_t weight_dim = weight.get_dim(1); // 每个token的嵌入向量的维度
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  // 单个句子最长512个token
  // hidden_units: 128
  constexpr int32_t block_num = 512;
  constexpr int32_t thread_num = 128;
  dim3 grid(block_num);
  dim3 block(thread_num);

  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    embedding_kernel<<<grid, block, 0, stream_>>>(
      vocab_size, input_num, weight_dim,
      input_cu.ptr<int32_t>(), weight.ptr<float>(), output.ptr<float>());
  } {
    embedding_kernel<<<grid, block>>>(
      vocab_size, input_num, weight_dim,
      input_cu.ptr<int32_t>(), weight.ptr<float>(), output.ptr<float>());
  }
}
} // namsespace kernel
