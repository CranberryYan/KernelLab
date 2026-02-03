add_executable(vec_add vec_add.cu)
target_link_libraries(vec_add PRIVATE ${CUDA_LIBRARIES})