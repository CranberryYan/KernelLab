add_executable(hello_world hello_world.cu)
target_link_libraries(hello_world PRIVATE ${CUDA_LIBRARIES})