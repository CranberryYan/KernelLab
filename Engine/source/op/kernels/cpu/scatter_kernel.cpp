#include "base/base.h"
#include "base/para.h"
#include "scatter_kernel.h"

namespace kernel {
void scatter_kernel_cpu(const tensor::Tensor &input,
                        const tensor::Tensor &src,
                        const tensor::Tensor &index,
                        tensor::Tensor &output,
                        para::scatter_para para,
                        void* stream) {
  uint32_t rows = input.get_dim(0);
  uint32_t input_cols = input.get_dim(1);
  uint32_t index_cols = index.get_dim(1);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < index_cols; ++col) {
      uint32_t index_offset = row * index_cols + col;
      int idx = index.at<int>(index_offset) < 0 ?
                index.at<int>(index_offset) + input_cols :
                index.at<int>(index_offset);
      if (idx >= 0 && idx < static_cast<int>(input_cols)) {
        uint32_t output_offset = row * input_cols + idx;
        if (para.op_type == para::ScatterOpType::Update) {
          output.at<float>(output_offset) = src.at<float>(index_offset);
        } else if (para.op_type == para::ScatterOpType::Add) {
          output.at<float>(output_offset) += src.at<float>(index_offset);
        } else {
          printf("Unknown para.op_type\n");
          return;
        }
      } else {
        printf("Index out of bounds at (%u, %u): idx = %d\n", row, col, idx);
      }
    }
  }
}
} // namespace kernel
