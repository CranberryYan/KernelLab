#ifndef KERNELLAB_INCLUDE_OP_HISTOGRAM_H_
#define KERNELLAB_INCLUDE_OP_HISTOGRAM_H_
#include "base/base.h"
#include "layer.h"

namespace op {
class HistogramLayer : public Layer {
public:
  explicit HistogramLayer(int32_t low_, int32_t high_,
                          base::DeviceType device_type);

  base::Status checkArgs() const override;

  base::Status compute() override;

  base::Status forward() override;

public:
  int32_t low;
  int32_t high;
};
} // namespace op
#endif // KERNELLAB_INCLUDE_OP_HISTOGRAM_H_