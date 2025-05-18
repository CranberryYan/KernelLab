
add_executable(hgemm_v1_wmma_m16n16k16_naive_kernel hgemm_v1_wmma_m16n16k16_naive_kernel.cu)
target_link_libraries(hgemm_v1_wmma_m16n16k16_naive_kernel PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hgemm_v1_wmma_m16n16k16_naive_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
# target_compile_options(hgemm_wmma_m16n16k16_naive_kernel PRIVATE -lineinfo)

add_executable(hgemm_v2_wmma_m16n16k16_mma4x2_kernel hgemm_v2_wmma_m16n16k16_mma4x2_kernel.cu)
target_link_libraries(hgemm_v2_wmma_m16n16k16_mma4x2_kernel PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hgemm_v2_wmma_m16n16k16_mma4x2_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel.cu)
target_link_libraries(hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
