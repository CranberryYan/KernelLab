#ifndef KERNELLAB_INCLUDE_OP_ADD_H_
#define KERNELLAB_INCLUDE_OP_ADD_H_
#include "base/base.h"
#include "layer.h"

namespace op {
class VecAddLayer : public Layer {
public:
  explicit VecAddLayer(base::DeviceType device_type);

  base::Status checkArgs() const override;

  base::Status compute() override;

  base::Status forward() override;
};
} // namespace op
#endif // KERNELLAB_INCLUDE_OP_ADD_H_
