#include <utility>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include <sentencepiece_processor.h>
#include "base/tick.h"
#include "op/matmul.h"
#include "op/rmsnorm.h"
#include "model/llama3.h"

namespace model {
void Llama2Model::attention_qkv(int32_t layer_idx,
                                const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);

  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  const auto &[key, val] =
    slice_kv_cache(layer_idx, pos);

  // query
  const std::shared_ptr<op::Layer> &query_layer =
    llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr)
    << "The query layer in the attention block is null pointer.";

  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key
  const std::shared_ptr<op::Layer> &key_layer =
    llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr)
    << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

  // value
  const std::shared_ptr<op::Layer> &value_layer =
    llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr)
    << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(llama_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(llama_layers_->rope_layer_->forward(query, key, pos_tensor,
    get_buffer(ModelBufferType::kSinCache),
    get_buffer(ModelBufferType::kCosCache),
    tensor::Tensor{}));
}
} // namespace model
