cmake_minimum_required(VERSION 3.18)

project(cuda_tutorial LANGUAGES CXX CUDA)

# ========== 基本编译设置 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 可选：默认 Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ========== CUDA 架构设置 ==========
# 默认用本机 GPU（CMake >= 3.18 支持 native）
# 你也可以手动：-DCMAKE_CUDA_ARCHITECTURES=86 / 89 / "86;89"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# ========== 找 CUDA Toolkit ==========
find_package(CUDAToolkit REQUIRED)

# ========== 小工具函数：添加一个 .cu 可执行文件 ==========
function(add_cuda_exe target src)
  add_executable(${target} ${src})

  target_link_libraries(${target} PRIVATE CUDA::cudart)
  target_compile_features(${target} PRIVATE cxx_std_17)

  # 让每个 target 显式继承架构（更稳）
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
  )

  # 可选：更好调试/剖析
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
endfunction()

# ========== 子目录 ==========
add_subdirectory(course1)
add_subdirectory(course2)
add_subdirectory(course3)
add_subdirectory(course4)
add_subdirectory(course6)
