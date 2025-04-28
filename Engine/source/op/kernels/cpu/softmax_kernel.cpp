#include "softmax_kernel.h"
#include "../kernels_interface.h"

namespace kernel {
void softmax_kernel_cpu(const tensor::Tensor& input,
                        tensor::Tensor& output,
                        para::softmax_para para,
                        void* stream) {
  uint32_t rows = input.get_dim(0);
  uint32_t cols = input.get_dim(1);
  std::vector<float> max(rows, -INFINITY);
  std::vector<float> sum(rows, 0.0f);

  if (para.op_type == para::SoftmaxOpType::Naive) {
    std::vector<float> input_tmp(rows*cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        uint32_t input_offset = r * cols + c;
        input_tmp[input_offset] = std::exp(input.at<float>(input_offset));
        sum[r] += input_tmp[input_offset];
      }
      for (int c = 0; c < cols; ++c) {
        uint32_t input_offset = r * cols + c;
        output.at<float>(input_offset) = input_tmp[input_offset] / sum[r];
      }
    }
  } else if (para.op_type == para::SoftmaxOpType::Safe) {
    std::vector<float> input_tmp(rows*cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        uint32_t input_offset = r * cols + c;
        max[r] = std::max(max[r], input.at<float>(input_offset));
      }
    }
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        uint32_t input_offset = r * cols + c;
        input_tmp[input_offset] =
          std::exp(input.at<float>(input_offset) - max[r]);
        sum[r] += input_tmp[input_offset];
      }
      for (int c = 0; c < cols; ++c) {
        uint32_t input_offset = r * cols + c;
        output.at<float>(input_offset) = input_tmp[input_offset] / sum[r];
      }
    }
  } else if (para.op_type == para::SoftmaxOpType::Online) {
    // 推导公式见notability
    for (int r = 0; r < rows; ++r) {
      float pre_max_tmp = 0;
      for (int c = 0; c < cols; ++c) {
        uint32_t input_offset = r * cols + c;
        max[r] = std::max(max[r], input.at<float>(input_offset));
        sum[r] =
          sum[r] * std::exp(pre_max_tmp - max[r]) +
          std::exp(input.at<float>(input_offset) - max[r]);
        pre_max_tmp = max[r];
      }
    }
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        uint32_t input_offset = r * cols + c;
        output.at<float>(input_offset) =
          std::exp(input.at<float>(input_offset) - max[r]) / sum[r];
      }
    }
  } else {
    printf("Unknown para.op_type\n");
    return;
  }
}
} // namespace kernel
