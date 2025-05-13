#ifndef KERNELLAB_INCLUDE_OP_REDUCE_H_
#define KERNELLAB_INCLUDE_OP_REDUCE_H_
#include "layer.h"
#include "base/base.h"

namespace op {
class ReduceLayer : public Layer {
public:
  explicit ReduceLayer(base::DeviceType device_type);

  base::Status checkArgs() const override;

  base::Status compute() override;

  base::Status forward() override;
};
} // namespace op
#endif // KERNELLAB_INCLUDE_OP_REDUCE_H_
