#include "base/API_trace.h"

namespace api_trace {
API_trace::API_trace(std::string func) : func_(func){
  printf("enter %s\n", func.c_str());
}
API_trace::~API_trace() { }

void API_trace::set_tensor(std::string name, tensor::Tensor tensor) {
  tensor_map[name] = tensor;
}

const char* API_trace::DeviceTypeToStr(base::DeviceType type) {
  switch (type) {
      case base::DeviceType::kDeviceUnknown:  return "kDeviceUnknown";
      case base::DeviceType::kDeviceCPU:      return "kDeviceCPU";
      case base::DeviceType::kDeviceCUDA:     return "kDeviceCUDA";
      default:                                return "UnknownDeviceType";
  }
}

const char* API_trace::DataTypeToStr(base::DataType type) {
  switch (type) {
      case base::DataType::kDataTypeInt32:   return "kDataTypeInt32";
      case base::DataType::kDataTypeFp32 :   return "kDataTypeFp32";
      case base::DataType::kDataTypeInt8 :   return "kDataTypeInt8";
      default:                               return "UnknownDeviceType";
  }
}

void API_trace::print_tensor(std::string name, tensor::Tensor ten) {
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto duration = now.time_since_epoch();
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() % 1000000;

  std::ostringstream time_stream;
  time_stream << std::put_time(std::localtime(&time_t_now), "%Y-%m-%dT%H:%M:%S")
              << "." << std::setw(6) << std::setfill('0') << micros;

  pid_t pid = getpid();
  std::ostringstream thread_id_stream;
  thread_id_stream << std::this_thread::get_id();
  std::string thread_id = thread_id_stream.str();

  printf("%s\n", name.c_str());

  printf("mem_handle: value=%p;\n", ten.get_buffer().get());

  printf(" deviceType: %s;\n", DeviceTypeToStr(ten.device_type()));
  printf(" dataType: %s;\n", DataTypeToStr(ten.data_type()));
  printf(" rank: %ld;\n", ten.dims().size());

  printf(" dims: [");
  for (int dim = 0; dim < ten.dims().size(); ++dim) {
      printf("%d", ten.get_dim(dim));
      if (dim != ten.dims().size() - 1) {
          printf(", ");
      }
  }
  printf("];\n");

  printf("Time: %s (0d+0h+0m+0s since start)\n", time_stream.str().c_str());
  printf("Process=%d;Thread=%s;StreamId=%p.\n",
          pid, thread_id.c_str(), ten.get_buffer().get());
  printf("\n");
}

void API_trace::print_tensor() {
  for (const auto& [key, val] : tensor_map) {
    this->print_tensor(key, val);
  }
}
} // api_trace
