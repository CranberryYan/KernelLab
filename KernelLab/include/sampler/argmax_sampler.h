#ifndef KERNELLAB_ARG_MAX_SAMPLER_H
#define KERNELLAB_ARG_MAX_SAMPLER_H
#include "sampler.h"
#include "base/base.h"

namespace sampler {
class ArgMaxSampler : public Sampler {
public:
  explicit ArgMaxSampler(base::DeviceType device_type) :
    Sampler(device_type) { }

  size_t sample(const float* logits, size_t size, void* stream) override;
};
} // namespace sampler
#endif  // KERNELLAB_NON_SAMPLER_H
