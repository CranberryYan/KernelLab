#ifndef ENGINE_INCLUDE_API_TRACE_H_
#define ENGINE_INCLUDE_API_TRACE_H_
#include <thread>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <unistd.h>

#include "base.h"
#include "../tensor/tensor.h"

namespace api_trace {
class API_trace {
public:
  API_trace(std::string func_);
  ~API_trace();

  void set_tensor(std::string name, tensor::Tensor tensor);

  const char* DeviceTypeToStr(base::DeviceType type);

  const char* DataTypeToStr(base::DataType type);

  void print_tensor(std::string name, tensor::Tensor ten);

  void print_tensor();
private:
  std::string func_;
  std::map<std::string, tensor::Tensor> tensor_map;
};
} // namespace api_trace
#endif // ENGINE_INCLUDE_API_TRACE_H_