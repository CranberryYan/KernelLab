#ifndef ENGINE_INCLUDE_OP_SWIGLU_H_
#define ENGINE_INCLUDE_OP_SWIGLU_H_
#include "layer.h"
namespace op {
class SwiGLULayer : public op::Layer {
public:
  explicit SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim);

  base::Status checkArgs() const override;

  base::Status compute() override;

  base::Status forward() override;

private:
  int32_t hidden_dim_ = 0;
};
} // namespace op
#endif // LLAMA_INFER_INCLUDE_OP_SWIGLU_H_