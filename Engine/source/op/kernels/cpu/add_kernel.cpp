#include "base/base.h"
#include "add_kernel.h"

namespace kernel
{
void add_kernel_cpu(const tensor::Tensor& input1,
                    const tensor::Tensor& input2,
                    tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  CHECK_EQ(input1.size(), input2.size());
  CHECK_EQ(input1.size(), output.size());

  for (int i = 0; i < input1.size(); ++i) {
    output.index<float>(i) = input1.index<float>(i) + input2.index<float>(i);
  }
}
} // namespace kernel