#include <cuda_runtime_api.h>
#include "base/alloc.h"

namespace base {
#define BigSize 1024 * 1024
#define MaxFreeSize 1024 * 1024 * 1024

#define DEBUG 0

void* CUDADeviceAllocator::allocate(size_t total_size) const {
  int id =  0;
  cudaError_t state = cudaGetDevice(&id); // 获取当前设备id
  CHECK(state == cudaSuccess);
  if (total_size > BigSize) {
    std::vector<CudaMemoryBuffer>& big_buffers = big_buffers_map_[id];
#if DEBUG
    printf("=============== enter here big_buffers_map_.size(): %ld\n",
      big_buffers_map_.size());
# endif
    // 优先使用已存在且未被占用的缓冲区
    //  big_buffers是否busy
    //  big_buffers的total_size(src)是否大于等于total_size(dst)
    //  big_buffers的total_size(src) - total_size(dst)是否小于BigSize(防止浪费)
    int sel_id = -1;
    for (int i = 0; i < big_buffers.size(); ++i) {
      if (!big_buffers[i].busy &&
          big_buffers[i].total_size >= total_size &&
          big_buffers[i].total_size - total_size < BigSize) {
        // 满足上述条件, 选择为缓冲区, 记录sel_id
        // ||: sel_id == -1 满足后, 不会运行big_buffers[sel_id] -> 没有索引问题
        if (sel_id == -1 ||
          big_buffers[sel_id].total_size > big_buffers[i].total_size) {
          sel_id = i;
        }
      }
    }
    // 使用已存在的缓冲区(找到了符合要求的buffer)
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
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
      LOG(ERROR) << buf;
      return nullptr;
    }

    // 将新分配的buffer加入big_buffers
    // CudaMemoryBuffer(void* data, size_t total_size, bool busy)
    big_buffers.emplace_back(ptr, total_size, true);
    return ptr;
  } else {
    std::vector<CudaMemoryBuffer>& cuda_buffers = cuda_buffers_map_[id];
#if DEBUG
    printf("=============== enter here cuda_buffers_map_.size(): %ld\n",
      cuda_buffers_map_.size());
# endif

    for (int i = 0; i < cuda_buffers.size(); ++i) {
      // 使用已存在的缓冲区
      if (cuda_buffers[i].total_size >= total_size &&
          !cuda_buffers[i].busy) {
        cuda_buffers[i].busy = true;
        no_busy_cnt_[id] -= cuda_buffers[i].total_size;
        return cuda_buffers[i].data;
      }
    }

    // 自行分配新的缓冲区(没找到符合要求的buffer)
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, total_size);
    if (cudaSuccess != state) {
      char buf[256];
      snprintf(buf, 256,
        "Error: CUDA error when allocating %lu MB memory! \
          maybe there's no enough memory "
        "left on device.", total_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }

    // 将新分配的buffer加入cuda_buffers
    cuda_buffers.emplace_back(ptr, total_size, true);
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
      std::vector<CudaMemoryBuffer> cuda_buffers = iter.second;
      std::vector<CudaMemoryBuffer> temp;
      for (int i = 0; i < cuda_buffers.size(); ++i) {
        if (!cuda_buffers[i].busy) {
          state = cudaSetDevice(iter.first);
          state = cudaFree(cuda_buffers[i].data);
          CHECK(state == cudaSuccess)
            << "Error: CUDA error when release memory on device " << iter.first;
        } else {
          temp.push_back(cuda_buffers[i]);
        }
      }
      cuda_buffers.clear();
      iter.second = temp;
      no_busy_cnt_[iter.first] = 0;
    }
  }

  // 释放指定buffer
  for (auto& iter : cuda_buffers_map_) {
    // 不是真的free, 只是将busy置为false, 并记录no_busy_cnt_
    std::vector<CudaMemoryBuffer> cuda_buffers = iter.second;
    for (int i = 0; i < cuda_buffers.size(); ++i) {
      if (cuda_buffers[i].data == ptr) {
        no_busy_cnt_[iter.first] += cuda_buffers[i].total_size;
        cuda_buffers[i].busy = false;
        return;
      }
    }
    std::vector<CudaMemoryBuffer> big_buffers = big_buffers_map_[iter.first];
    for (int i = 0; i < big_buffers.size(); ++i) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return;
      }
    }
  }

  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) <<
    "Error: CUDA error when release memory on device";
}

std::shared_ptr<CUDADeviceAllocator>
  CUDADeviceAllocatorFactory::instance = nullptr;
} // namespace base
