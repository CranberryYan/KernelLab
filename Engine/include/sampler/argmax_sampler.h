#ifndef LLAMA_INFER_ARG_MAX_SAMPLER_H
#define LLAMA_INFER_ARG_MAX_SAMPLER_H
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
#endif  // LLAMA_INFER_NON_SAMPLER_H
