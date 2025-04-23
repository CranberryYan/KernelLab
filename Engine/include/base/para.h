#ifndef ENGINE_INCLUDE_PARA_H_
#define ENGINE_INCLUDE_PARA_H_

namespace para {
struct add_para {
  uint32_t bpe;
  uint32_t ele_num;
  uint32_t size;

  uint32_t block_num;
  uint32_t thread_num;
};

struct reduce_para {
  uint32_t bpe;
  uint32_t ele_num;
  uint32_t after_reduce_num;
  uint32_t size;

  uint32_t block_num;
  uint32_t thread_num;
};
} // namespace para
#endif // ENGINE_INCLUDE_PARA_H_