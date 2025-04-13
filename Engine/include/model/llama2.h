#ifndef ENGINE_INCLUDE_MODEL_LLAMA_H_
#define ENGINE_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/rope.h"
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

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;

  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class Llama2Model : public Model {
public:
explicit Llama2Model(base::TokenizerType tokenizer_type,
                     std::string token_path,
                     std::string model_path,
                     bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input,
                       const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input,
                       const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(std::vector<int>& tokens) override;

private:
  void init_mem() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  base::Status create_layers() override;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos,
                          bool is_prompt) const override;

private:
  std::unique_ptr<Llama2Layers> llama_layers_;
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
};
} // namespace model
#endif // ENGINE_INCLUDE_MODEL_LLAMA_H_
