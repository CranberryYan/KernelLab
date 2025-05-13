#ifndef KERNELLAB_INCLUDE_OP_SCATTER_H_
#define KERNELLAB_INCLUDE_OP_SCATTER_H_
#include "base/base.h"
#include "base/para.h"
#include "layer.h"

namespace op {
class ScatterLayer : public Layer {
public:
  explicit ScatterLayer(base::DeviceType device_type);

  base::Status checkArgs() const override;

  base::Status compute() override;

  base::Status forward() override;

public:
  para::ScatterOpType op_type = para::ScatterOpType::Scatter_Update;
};
}
#endif // KERNELLAB_INCLUDE_OP_SCATTER_H_
