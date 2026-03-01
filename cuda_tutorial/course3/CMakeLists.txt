add_executable(softmax softmax.cu)
add_executable(sulf_sync shuffle_sync.cu)
add_executable(sulf_up shuffle_up.cu)
add_executable(sulf_down shuffle_down.cu)

target_link_libraries(softmax PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(sulf_sync PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(sulf_up PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(sulf_down PRIVATE ${CUDA_LIBRARIES})
