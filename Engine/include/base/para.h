#ifndef ENGINE_INCLUDE_PARA_H_
#define ENGINE_INCLUDE_PARA_H_
#include <iostream>
#include <vector>

#define SMEM_ALIGN 128

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
  Scatter_Add = 0,
  Scatter_Update = 1,
  Gather = 2
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

enum class SoftmaxOpType {
  Naive = 0,
  Safe = 1,
  Online = 2
};

struct softmax_para : public para_base {
  std::vector<int32_t> input_dims = std::vector<int32_t>(2);
  std::vector<int32_t> output_dims = std::vector<int32_t>(2);

  uint32_t input_cols;
  uint32_t input_rows;

  SoftmaxOpType op_type;
};

enum class EnflameDevice {
  GCU300 = 0,
  GCU400 = 1
};

struct index_add_para : public para_base {
  std::vector<int32_t> target_dims = std::vector<int32_t>(2);
  std::vector<int32_t> index_dims = std::vector<int32_t>(1);
  std::vector<int32_t> source_dims = std::vector<int32_t>(2);

  uint32_t target_rows;
  uint32_t target_cols;
  uint32_t index_nums;
  uint32_t source_rows;
  uint32_t source_cols;

  uint32_t bpe;
  uint32_t ele_num;
  uint32_t size;

  uint32_t ele_num_per_block;
  uint32_t block_per_target_row;
  uint32_t block_per_source_row;

  EnflameDevice enflame_device;
};
} // namespace para
#endif // ENGINE_INCLUDE_PARA_H_
