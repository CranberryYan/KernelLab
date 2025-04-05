#ifndef ENGINE_INCLUDE_OP_RMSNORM_H_
#define ENGINE_INCLUDE_OP_RMSNORM_H_
#include "layer.h"
namespace op {
class RMSNormLayer : public LayerParam {
public:
  explicit RMSNormLayer(base::DeviceType device_type, int32_t dim);

  base::Status checkArgs() const override;

  base::Status forward() override;

private:
  int32_t dim_ = 0;
};
} // namespace op
#endif // ENGINE_INCLUDE_OP_RMSNORM_H_
