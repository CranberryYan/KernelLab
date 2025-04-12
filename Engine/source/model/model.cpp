#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "model/model.h"

namespace model {
Model::Model(base::TokenizerType tokenizer_type,
             base::ModelType model_type,
             std::string token_path,
             std::string model_path,
             bool is_quant_model) :
  tokenizer_type_(tokenizer_type),
  model_type_(model_type),
  token_path_(std::move(token_path)),
  model_path_(std::move(model_path)),
  is_quant_model_(is_quant_model) { }

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

// get_buffer:
//  错误检查: CHECK_GT: 当前buffer_idx(key)对应val个数必须大于0
//  返回: key所对应的val
tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(
  int32_t layer_idx, int32_t token_pos) const {
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(
        get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(
        get_buffer(ModelBufferType::kValCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32,
                     config_->kv_dim_,
                     false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32,
                     config_->kv_dim_,
                     false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);

  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }

  // 根据当前 token 的位置, 从 embedding 输出中提取该 token 对应的嵌入向量
  std::shared_ptr<base::Buffer> input_embedding_buffer =
    std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr,
      input_embeddings.ptr<float>(index * config_->dim_), true);
  
  tensor::Tensor input(base::DataType::kDataTypeFp32,
                       config_->dim_, false, nullptr, nullptr);

  input.assign(input_embedding_buffer);
  input.set_device_type(device_type_);
  return input;
}

base::Status Model::insert_buffer(ModelBufferType buffer_idx,
                                  tensor::Tensor &tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(
      std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }

  if (tensor.is_empty()) {
    return base::error::InvalidArgument(
      "The tensor is empty for inserting buffer.");
  }

  buffers_.insert({buffer_idx, tensor});

  return base::error::Success();
}

base::Status Model::read_model_file() {
  if (model_path_.empty()) {
    return base::error::PathNotValid(
      "Failed to open the weight file, the model path is empty!");
  }

  int fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return base::error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return base::error::PathNotValid(
      "Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig();

  // 从 file 中读取 sizeof(ModelConfig) 大小的数据并存入 config
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return base::error::ModelParseError(
      "Failed to retrieve the configuration information from the model file.");
  }

  if (is_quant_model_) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      return base::error::ModelParseError(
        "Failed to retrieve the group size information from the model file.");
    }
  }

  base::Status gen_status = generate_model_infos(config);
  if (!gen_status) {
    return gen_status;
  }

  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd);
    return base::error::ModelParseError(
        "Failed to retrieve the file size information from the model file.");
  }

  raw_model_data_->file_size = st.st_size;
  LOG(INFO) << "The tokenizer model path: " << token_path_;
  std::string tokenizer_type_str = tokenizer_type_ == base::TokenizerType::kEncodeBpe
                                   ? "Bpe" : "Spe";
  LOG(INFO) << "The tokenizer type: " << tokenizer_type_str;

  LOG(INFO) << "The model path: " << model_path_;
  LOG(INFO) << "The model file size: " << raw_model_data_->file_size << " byte";
  std::string quant_info = is_quant_model_ ? "quant" : "not quant";
  LOG(INFO) << "The model is " << quant_info << " model";

  if (config_) {
    LOG(INFO) << "\nThe model info: " << *config_;
  }

  raw_model_data_->fd = fd;
  raw_model_data_->data = mmap(nullptr, raw_model_data_->file_size,
                               PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);
  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return base::error::ModelParseError(
      "Failed to map the weight file " + model_path_ + " into memory.");
  }

  if (!is_quant_model_) {
    raw_model_data_->weight_data =
      static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
  } else {
    raw_model_data_->weight_data =
      static_cast<int8_t*>(raw_model_data_->data) +
      sizeof(ModelConfig) + sizeof(group_size_);
  }

  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return base::error::ModelParseError(
      "Failed to map the weight file " + model_path_ +
      " into memory, the pointer to weight start address is null");
  }

  return base::error::Success();
}

base::Status Model::create_encode_layer() {
  // create token encode and decode layer
  if (tokenizer_type_ == base::TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(
      this->token_path_, true, false);
    if (!encode_layer_) {
      return base::error::InternalError("Create the encode layer failed.");
    }
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return base::error::InternalError("The vocab size param read error from the model file!");
  }

  return base::error::Success();
}

base::Status Model::gen_model_from_file() {
  config_ = std::make_unique<TransformerConfig>();
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed! " << create_encode_status.get_err_msg();
    return create_encode_status;
  }

  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) 
      << "Read model file " 
      << model_path_ << " failed! " 
      << mmap_status.get_err_msg();
    return mmap_status;
  }

  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " 
               << model_path_ << " failed! "
               << mmap_status.get_err_msg();
    return layer_create_status;
  }

  return base::error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig &config) const {
  config_->dim_         = config.dim;
  config_->seq_len_     = config.seq_len;
  config_->layer_num_   = config.layer_num;
  config_->hidden_dim_  = config.hidden_dim;

  config_->head_num_    = config.head_num;
  config_->head_size_   = config.dim / config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->kv_mul_      = config.head_num / config.kv_head_num;
  config_->kv_dim_      = (config.dim * config.kv_head_num) / config.head_num;

  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }
  config_->vocab_size_ = std::abs(config.vocab_size);

  return base::error::Success();
}
} // namespace model
