#ifndef ENGINE_INCLUDE_OP_INDEX_ADD_H_
#define ENGINE_INCLUDE_OP_INDEX_ADD_H_
#include "base/base.h"
#include "layer.h"

namespace op {
class IndexAddLayer : public Layer {
public:
  explicit IndexAddLayer(base::DeviceType device_type);

  base::Status checkArgs() const override;

  base::Status compute() override;

  base::Status forward() override;
public:
  para::EnflameDevice enflame_device = para::EnflameDevice::GCU300;
};
} // namespace op
#endif // ENGINE_INCLUDE_OP_INDEX_ADD_H_