add_executable(hello_world hello_world.cu)
target_link_libraries(hello_world PRIVATE ${CUDA_LIBRARIES})

target_compile_options(hello_world PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
)