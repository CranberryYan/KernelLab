#ifndef ENGINE_INCLUDE_MODEL_MODEL_H_
#define ENGINE_INCLUDE_MODEL_MODEL_H_
#include <map>
#include <string>
#include <sentencepiece_processor.h>
#include "config.h"
#include "op/layer.h"
#include "op/embedding.h"
#include "tensor/tensor.h"
namespace model {
class Model {
public:
  explicit Model(base::TokenizerType tokenizer_type,
                 base::ModelType model_type,
                 std::string token_path,
                 std::string model_path,
                 bool is_quant_model);

  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);
  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(
    int32_t layer_idx, int32_t token_pos) const;

protected:
  int32_t group_size_ = 1;
  bool is_quant_model_ = false;
  std::unique_ptr<TransformerConfig> config_;

  std::string token_path_;
  std::string model_path_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;
};
} // namespace mode 
#endif // ENGINE_INCLUDE_MODEL_MODEL_H_