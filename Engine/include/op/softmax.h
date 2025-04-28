#ifndef ENGINE_INCLUDE_OP_SOTFMAX_H_
#define ENGINE_INCLUDE_OP_SOTFMAX_H_
#include "base/base.h"
#include "base/para.h"
#include "layer.h"

namespace op {
class SoftmaxLayer : public Layer {
public:
  explicit SoftmaxLayer(base::DeviceType device_type);

  base::Status checkArgs() const override;

  base::Status compute() override;

  base::Status forward() override;
public:
  para::SoftmaxOpType op_type = para::SoftmaxOpType::Naive;
};
} // namespace op
#endif // ENGINE_INCLUDE_OP_SOTFMAX_H_