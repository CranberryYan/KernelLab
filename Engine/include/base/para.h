#ifndef ENGINE_INCLUDE_PARA_H_
#define ENGINE_INCLUDE_PARA_H_
#include <iostream>
#include <vector>

namespace para {
struct para_base {

};

struct add_para : public para_base {
  uint32_t bpe;
  uint32_t ele_num;
  uint32_t size;

  uint32_t block_num;
  uint32_t thread_num;
};

struct reduce_para : public para_base {
  uint32_t bpe;
  uint32_t ele_num;
  uint32_t after_reduce_num;
  uint32_t size;

  uint32_t block_num;
  uint32_t thread_num;
};

enum class ScatterOpType {
  Add = 0,
  Update = 1
};

struct scatter_para : public para_base {
  std::vector<int32_t> input_dims = std::vector<int32_t>(2);
  std::vector<int32_t> index_dims = std::vector<int32_t>(2);
  std::vector<int32_t> src_dims = std::vector<int32_t>(2);
  uint32_t input_cols;
  uint32_t input_rows;

  uint32_t bpe;
  uint32_t index_ele_num;
  uint32_t input_ele_num;
  uint32_t src_ele_num;
  uint32_t index_ele_num_per_block;
  uint32_t input_ele_num_per_block;
  uint32_t src_ele_num_per_block;

  uint32_t size;

  uint32_t block_num;
  uint32_t thread_num;

  ScatterOpType op_type;
};
} // namespace para
#endif // ENGINE_INCLUDE_PARA_H_
