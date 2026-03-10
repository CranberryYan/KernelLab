add_executable(histogram histogram.cu)
target_link_libraries(histogram PRIVATE ${CUDA_LIBRARIES})
