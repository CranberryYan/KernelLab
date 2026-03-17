#include <cstdio>
#include "base/alloc.h"
#include <cuda_runtime_api.h>

namespace base {
#define BigSize 1024 * 1024
#define MaxFreeSize 1024 * 1024 * 1024

#define DEBUG 0

void* CUDADeviceAllocator::allocate(size_t total_size) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id); // 获取当前设备id
  if (state != cudaSuccess) {
    fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(state));
    return nullptr;
  }
  if (total_size > BigSize) {
    auto& big_buffers = big_buffers_map_[id];
#if DEBUG
    printf("=============== enter here big_buffers_map_.size(): %ld\n",
      big_buffers_map_.size());
# endif
    // 优先使用已存在且未被占用的缓冲区
    //  big_buffers是否busy
    //  big_buffers的total_size(src)是否大于等于total_size(dst)
    //  big_buffers的total_size(src) - total_size(dst)是否小于BigSize(防止浪费)
    CudaMemoryBuffer* best = nullptr;
    for (auto& buffer : big_buffers) {
      if (!buffer.busy &&
          buffer.total_size >= total_size &&
          buffer.total_size - total_size < BigSize) {
        if (best == nullptr || best->total_size > buffer.total_size) {
          best = &buffer;
        }
      }
    }
    // 使用已存在的缓冲区(找到了符合要求的buffer)
    if (best != nullptr) {
      best->busy = true;
      return best->data;
    }

    // 分配新的缓冲区(没找到符合要求的buffer)
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, total_size);
    if (cudaSuccess != state) {
      char buf[256];
      snprintf(buf, 256,
        "Error: CUDA error when allocating %lu MB memory! \
          maybe there's no enough memory "
        "left on device.", total_size >> 20);
      fprintf(stderr, "%s\n", buf);
      return nullptr;
    }

    // 将新分配的buffer加入big_buffers
    // CudaMemoryBuffer(void* data, size_t total_size, bool busy)
    big_buffers.push_back(CudaMemoryBuffer(ptr, total_size, true));
    return ptr;
  } else {
    auto& cuda_buffers = cuda_buffers_map_[id];
#if DEBUG
    printf("=============== enter here cuda_buffers_map_.size(): %ld\n",
      cuda_buffers_map_.size());
# endif

    for (auto& buffer : cuda_buffers) {
      // 使用已存在的缓冲区
      if (buffer.total_size >= total_size &&
          !buffer.busy) {
        buffer.busy = true;
        no_busy_cnt_[id] -= buffer.total_size;
        return buffer.data;
      }
    }

    // 自行分配新的缓冲区(没找到符合要求的buffer)
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, total_size);
    if (cudaSuccess != state) {
      char buf[256];
      snprintf(buf, 256,
        "Error: CUDA error when allocating %lu MB memory! \
          maybe there's no enough memory left on device.", total_size >> 20);
      fprintf(stderr, "%s\n", buf);
      return nullptr;
    }

    // 将新分配的buffer加入cuda_buffers
    cuda_buffers.push_back(CudaMemoryBuffer(ptr, total_size, true));
    return ptr;
  }
}

void CUDADeviceAllocator::release(void* ptr) const {
  if (ptr == nullptr || cuda_buffers_map_.empty()) {
    return;
  }

  cudaError_t state = cudaSuccess;

  // 并不频繁的free, 等待空闲buffercnt达到MaxFreeSize后free
  for (auto& iter : cuda_buffers_map_) {
    if (no_busy_cnt_[iter.first] > MaxFreeSize) {
      auto& cuda_buffers = iter.second;
      for (auto it = cuda_buffers.begin(); it != cuda_buffers.end();) {
        if (!it->busy) {
          state = cudaSetDevice(iter.first);
          state = cudaFree(it->data);
          if (state != cudaSuccess) {
            fprintf(
              stderr, "Error: \
              CUDA error when release memory on device %d, reason: %s\n",
              iter.first, cudaGetErrorString(state));
          }
          it = cuda_buffers.erase(it);
        } else {
          ++it;
        }
      }
      no_busy_cnt_[iter.first] = 0;
    }
  }

  // 释放指定buffer
  for (auto& iter : cuda_buffers_map_) {
    // 不是真的free, 只是将busy置为false, 并记录no_busy_cnt_
    auto& cuda_buffers = iter.second;
    for (auto& buffer : cuda_buffers) {
      if (buffer.data == ptr) {
        no_busy_cnt_[iter.first] += buffer.total_size;
        buffer.busy = false;
        return;
      }
    }
    auto& big_buffers = big_buffers_map_[iter.first];
    for (auto& buffer : big_buffers) {
      if (buffer.data == ptr) {
        buffer.busy = false;
        return;
      }
    }
  }

  state = cudaFree(ptr);
  if (state != cudaSuccess) {
    fprintf(stderr, "Error: \
            CUDA error when release memory on device, reason: %s\n",
            cudaGetErrorString(state));
  }
}

std::shared_ptr<CUDADeviceAllocator>
  CUDADeviceAllocatorFactory::instance = nullptr;
} // namespace base
