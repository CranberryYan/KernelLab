#ifndef ENGINE_INCLUDE_MODEL_LLAMA_H_
#define ENGINE_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/swiglu.h"
#include "op/embedding.h"
namespace model {
struct Llama2Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  // attention
  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class Llama2Model : public Model {
public:
explicit Llama2Model(base::TokenizerType tokenizer_type,
                     std::string token_path,
                     std::string model_path,
                     bool is_quant_model);

private:
  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

private:
  std::unique_ptr<Llama2Layers> llama_layers_;
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
};
} // namespace model
#endif // ENGINE_INCLUDE_MODEL_LLAMA_H_