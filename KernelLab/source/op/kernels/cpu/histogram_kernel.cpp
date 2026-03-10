#include "base/base.h"
#include "histogram_kernel.h"

namespace kernel {
void histogram_kernel_cpu(const tensor::Tensor& input,
                          tensor::Tensor& output,
                          para::histogram_para para,
                          void* stream) {
  CHECK_EQ(input.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  int32_t low = para.low;
  int32_t high = para.high;
  int32_t length = high - low;
  int32_t ele_num = para.ele_num;

  for (int i = 0; i < ele_num; ++i) {
    int32_t val = input.at<int32_t>(i);
    if (val >= low && val < high) {
      int32_t cnt = output.at<int32_t>(val - low);
      cnt++;
      output.set_value<int32_t>(cnt, val - low);
    }
  }
}
} // namespace kernel
