#include "base/base.h"
#include "index_add_kernel.h"

namespace kernel {
void index_add_kernel_cpu(const tensor::Tensor& target,
                          const tensor::Tensor& index,
                          const tensor::Tensor& source,
                          tensor::Tensor& output,
                          para::index_add_para para,
                          void* stream) {
  CHECK_EQ(target.is_empty(), false);
  CHECK_EQ(index.is_empty(), false);
  CHECK_EQ(source.is_empty(), false);

  const uint32_t target_cols = target.get_dim(1);

  for (int i = 0; i < index.get_dim(0); ++i) {
    int32_t target_rows_used = index.at<int32_t>(i);
    for (int s = 0; s < target_cols; ++s) {
      float tmp =
        output.at<float>(target_rows_used * target_cols + s) +
        source.at<float>(i * target_cols + s);
      output.set_value<float>(tmp, target_rows_used * target_cols + s);
    }
  }
}
} // namespace kernel
