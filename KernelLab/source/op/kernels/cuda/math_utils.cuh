#ifndef MATH_UTILS_CUH_
#define MATH_UTILS_CUH_
#include <iostream>
#include <cuda_runtime_api.h>
namespace math_cu {
#define KERNEL_HOST_INLINE __host__ __device__ __forceinline__

template<typename T>
KERNEL_HOST_INLINE constexpr T CeilDiv(T x, T y) {
  return (x + y - 1) / y;
}

template<typename T>
KERNEL_HOST_INLINE constexpr T AlignUp(T x, T y) {
  return CeilDiv<T>(x, y) * y;
}

template<typename T>
KERNEL_HOST_INLINE constexpr T AlignDown(T x, T y) {
  return (x / y) * y;
}
} // namespace math_cu
#endif // MATH_UTILS_CUH_