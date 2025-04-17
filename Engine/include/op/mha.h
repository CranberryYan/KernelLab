#ifndef ENGINE_INCLUDE_OP_MHA_H_
#define ENGINE_INCLUDE_OP_MHA_H_
#include "layer.h"
#include <base/cuda_config.h>

namespace op {
class MHA : public Layer {
public:
  explicit MHA(base::DeviceType device_type, int32_t layer_index,
               int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
               int32_t head_num, int32_t head_size);

  base::Status checkArgs() const override;

  void set_pos(int32_t pos);
  void set_layer_idx(int32_t layer_idx);

  base::Status compute() override;

  base::Status forward() override;

private:
  int32_t pos_ = 0;
  int32_t kv_mul_ = 0;
  int32_t kv_dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t head_num_ = 0;
  int32_t head_size_ = 0;
  int32_t layer_index_ = 0;
};
} // namespace op
#endif // ENGINE_INCLUDE_OP_MHA_H_
