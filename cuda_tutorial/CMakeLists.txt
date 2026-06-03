cmake_minimum_required(VERSION 3.18)

project(cuda_tutorial LANGUAGES CXX CUDA)

# ========== 基本编译设置 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 生成 compile_commands.json，方便 IDE / clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 默认 Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ========== CUDA 架构设置 ==========
# 如果 native 报错，就手动指定：
#   RTX 3080: -DCMAKE_CUDA_ARCHITECTURES=86
#   RTX 4060: -DCMAKE_CUDA_ARCHITECTURES=89
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# ========== CUDA 编译选项开关 ==========
option(CUDA_PTXAS_VERBOSE "Print ptxas register/spill information" ON)
option(CUDA_USE_FAST_MATH "Enable --use_fast_math for CUDA" OFF)
option(CUDA_ENABLE_RDC "Enable CUDA separable compilation / RDC" OFF)

message(STATUS "CUDA_PTXAS_VERBOSE = ${CUDA_PTXAS_VERBOSE}")
message(STATUS "CUDA_USE_FAST_MATH = ${CUDA_USE_FAST_MATH}")
message(STATUS "CUDA_ENABLE_RDC = ${CUDA_ENABLE_RDC}")

# ========== 找 CUDA Toolkit ==========
find_package(CUDAToolkit REQUIRED)

add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

# ========== 小工具函数：添加一个 .cu 可执行文件 ==========
function(add_cuda_exe target src)
  add_executable(${target} ${src})

  target_link_libraries(${target} PRIVATE CUDA::cudart)
  target_compile_features(${target} PRIVATE cxx_std_17)

  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ${CUDA_ENABLE_RDC}
  )

  # CUDA 编译选项
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )

  # 打印寄存器 / spill 信息
  if(CUDA_PTXAS_VERBOSE)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
  endif()

  # 可选 fast math
  if(CUDA_USE_FAST_MATH)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
  endif()
endfunction()

# ========== 子目录 ==========
add_subdirectory(course1)
add_subdirectory(course2)
add_subdirectory(course3)
add_subdirectory(course4)
add_subdirectory(course5)
add_subdirectory(course6)
add_subdirectory(course13)
add_subdirectory(NewFeature)
