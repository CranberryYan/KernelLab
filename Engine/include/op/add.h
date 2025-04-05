#ifndef ENGINE_INCLUDE_OP_ADD_H_
#define ENGINE_INCLUDE_OP_ADD_H_
#include "base/base.h"
#include "layer.h"
namespace op {
class VecAddLayer : public Layer {
 public:
  explicit VecAddLayer(base::DeviceType device_type);

  base::Status checkArgs() const override;

  base::Status forward() override;
};
}  // namespace op
#endif  // ENGINE_INCLUDE_OP_ADD_H_
