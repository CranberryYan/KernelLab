add_executable(TensorCore_demo TensorCore.cu)
target_link_libraries(TensorCore_demo PRIVATE CUDA::cudart CUDA::cublas)
