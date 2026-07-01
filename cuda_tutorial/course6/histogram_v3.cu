#include <random>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>

#define BIN_GROUP 4096
#define GRID_SIZE 512
#define BLOCK_SIZE 512

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void histogram_v3(int* input, int* hist_blocks, int n,
                             int low, int high, int length,
                             int bin_start, int cur_bins) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int num_threads = blockDim.x;

  extern __shared__ int s_hist[];
  for (int i = tid; i < cur_bins; i += num_threads) {
    s_hist[i] = 0;
  }
  __syncthreads();

  // 只统计落在当前bin group的值
  int stride = gridDim.x * blockDim.x;
  for (int i = gid; i < n; i += stride) {
    int val = input[i];
    int bin = val - low;
    if (bin >= bin_start && bin < (bin_start + cur_bins)) {
      atomicAdd(&s_hist[bin - bin_start], 1);
    }
  }
  __syncthreads();

  for (int i = tid; i < cur_bins; i += num_threads) {
    hist_blocks[blockIdx.x * cur_bins + i] = s_hist[i];
  }
}

// ============================================================
// CPU 参考实现
// ============================================================
void cpu_histogram(const int* input, int* hist, int n, int low, int high) {
  int length = high - low;
  std::fill(hist, hist + length, 0);
  for (int i = 0; i < n; ++i) {
    int val = input[i];
    if (val >= low && val < high) {
      hist[val - low] += 1;
    }
  }
}

inline void checkCudaError(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::cerr << "[CUDA ERROR] " << msg << ": "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  struct TestConfig {
    int n;      // 输入数据个数
    int length; // 直方图 bin 个数 (high - low)
  };

  std::vector<TestConfig> configs = {
    // { n, length }
    {       16384,     256 },
    {      131072,     256 },
    {     1048576,     256 },
    {     4194304,     256 },
    {      131072,     512 },
    {      131072,    1024 },
    {      131072,    2048 },
    {      131072,    4096 },
    // v4: 大 bin 场景, 测试 multi-pass 的开销
    {      524288,   16384 },
    {      524288,   65536 },
  };

  const int low = -128;
  const int warmup_time = 10;   // 每个 bin pass 的 warmup 次数
  const int repeat_time = 1024;   // 计时重复次数 (v4 每次都含多个 kernel launch)

  const int grid_size = GRID_SIZE;
  const int block_size = BLOCK_SIZE;
  const size_t smem_bytes = BIN_GROUP * sizeof(int);

  size_t max_block_bytes = BIN_GROUP * GRID_SIZE * sizeof(int);

  // ---------- CSV 文件 ----------
  std::ofstream csv_file("./course6/histogram_v3_benchmark.csv");
  csv_file << "N,Length,BinPasses,GPU_Time_ms,Bandwidth_GBs,ErrorCount,Pass"
           << std::endl;

  std::cout << "========================================" << std::endl;
  std::cout << "histogram_v3 (Split-by-Length, No Global Atomic) Benchmark"
            << std::endl;
  std::cout << "========================================" << std::endl;

  for (const auto& cfg : configs) {
    int N = cfg.n;
    int length = cfg.length;
    int high = low + length;
    int num_bin_passes = CEIL_DIV(length, BIN_GROUP);

    std::cout << "\n--- Testing: N=" << N << ", bins=" << length
              << " (" << num_bin_passes << " pass(es)) ---" << std::endl;

    size_t input_bytes = N * sizeof(int);
    size_t hist_bytes = length * sizeof(int);
    int total_hist_sum = 0;

    // ---------- Host 内存 ----------
    int* h_input     = nullptr;
    int* h_hist_ref  = nullptr;
    int* h_hist_gpu  = nullptr;
    int* h_blocks    = nullptr;  // 接收 d_hist_blocks 的临时 buffer

    // ---------- Device 内存 ----------
    int *d_input       = nullptr;
    int *d_hist_blocks = nullptr;  // 各 block 私有输出的临时 buffer

    bool out_of_memory = false;

    try {
      h_input    = (int*)malloc(input_bytes);
      h_hist_ref = (int*)malloc(hist_bytes);
      h_hist_gpu = (int*)malloc(hist_bytes);
      h_blocks   = (int*)malloc(max_block_bytes);
      if (!h_input || !h_hist_ref || !h_hist_gpu || !h_blocks) {
        throw std::bad_alloc();
      }

      checkCudaError(cudaMalloc(&d_input, input_bytes),
                     "cudaMalloc d_input failed");
      checkCudaError(cudaMalloc(&d_hist_blocks, max_block_bytes),
                     "cudaMalloc d_hist_blocks failed");

      // ---------- 数据初始化 ----------
      std::mt19937 rng(12345);
      std::uniform_int_distribution<int> dist(low, high - 1);
      std::uniform_int_distribution<int> dist_oob(low - 500, high + 500);
      std::uniform_real_distribution<float> dist_oob_ratio(0.0f, 1.0f);

      for (int i = 0; i < N; ++i) {
        if (dist_oob_ratio(rng) < 0.1f) {
          h_input[i] = dist_oob(rng);
        } else {
          h_input[i] = dist(rng);
        }
      }

      // CPU 参考
      cpu_histogram(h_input, h_hist_ref, N, low, high);

      // 拷贝 input 到 device
      checkCudaError(cudaMemcpy(d_input, h_input, input_bytes,
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy d_input failed");

      // ---------- CUDA Events ----------
      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start)");
      checkCudaError(cudaEventCreate(&stop),  "cudaEventCreate(stop)");

      // ---------- Warmup: 跑一遍所有 bin pass ----------
      for (int w = 0; w < warmup_time; ++w) {
        for (int bin_start = 0; bin_start < length; bin_start += BIN_GROUP) {
          int bin_end = std::min(bin_start + BIN_GROUP, length);
          int cur_bins = bin_end - bin_start;
          histogram_v3<<<grid_size, block_size, cur_bins * sizeof(int)>>>(
              d_input, d_hist_blocks, N, low, high, length,
              bin_start, cur_bins);
        }
      }
      cudaDeviceSynchronize();
      checkCudaError(cudaGetLastError(), "Warmup kernel failed");

      // ---------- 正式计时：所有 bin pass 作为一次完整运行 ----------
      float total_time_ms = 0.0f;
      for (int r = 0; r < repeat_time; ++r) {
        checkCudaError(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int bin_start = 0; bin_start < length; bin_start += BIN_GROUP) {
          int bin_end = std::min(bin_start + BIN_GROUP, length);
          int cur_bins = bin_end - bin_start;
          histogram_v3<<<grid_size, block_size, cur_bins * sizeof(int)>>>(
              d_input, d_hist_blocks, N, low, high, length,
              bin_start, cur_bins);
        }
        checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop)");
        checkCudaError(cudaEventSynchronize(stop),
                       "cudaEventSynchronize(stop)");
        float elapsed_ms = 0.0f;
        checkCudaError(cudaEventElapsedTime(&elapsed_ms, start, stop),
                       "cudaEventElapsedTime");
        total_time_ms += elapsed_ms;
      }
      float avg_time_ms = total_time_ms / repeat_time;

      // ---------- 最后一次运行 + CPU reduce ----------
      memset(h_hist_gpu, 0, hist_bytes);
      for (int bin_start = 0; bin_start < length; bin_start += BIN_GROUP) {
        int bin_end = std::min(bin_start + BIN_GROUP, length);
        int cur_bins = bin_end - bin_start;

        histogram_v3<<<grid_size, block_size, cur_bins * sizeof(int)>>>(
            d_input, d_hist_blocks, N, low, high, length,
            bin_start, cur_bins);
        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "Final kernel failed");

        // 拷贝 per-block 结果
        size_t block_bytes = cur_bins * GRID_SIZE * sizeof(int);
        checkCudaError(
            cudaMemcpy(h_blocks, d_hist_blocks, block_bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy d_hist_blocks failed");

        // CPU reduce: 对每个 bin, sum across all blocks
        for (int bid = 0; bid < GRID_SIZE; ++bid) {
          for (int i = 0; i < cur_bins; ++i) {
            h_hist_gpu[bin_start + i] += h_blocks[bid * cur_bins + i];
          }
        }
      }

      // ---------- 正确性验证 ----------
      int error_count = 0;
      const int max_errors_to_print = 10;
      for (int i = 0; i < length; ++i) {
        if (h_hist_gpu[i] != h_hist_ref[i]) {
          ++error_count;
          if (error_count <= max_errors_to_print) {
            std::cout << "  Mismatch[" << i << "]: ref=" << h_hist_ref[i]
                      << ", gpu=" << h_hist_gpu[i] << std::endl;
          }
        }
        total_hist_sum += h_hist_gpu[i];
      }
      if (error_count > max_errors_to_print) {
        std::cout << "  ... (" << (error_count - max_errors_to_print)
                  << " more errors)" << std::endl;
      }

      bool passed = (error_count == 0);

      // 带宽 = 总读取量 / 时间 (重复读取计入)
      // 每次 input 被读了 num_bin_passes 次
      float total_bytes_read = 1.0f * input_bytes * num_bin_passes;
      float bandwidth_gbs = (total_bytes_read / (avg_time_ms * 1e6f));

      std::cout << "  [Result] avg_time=" << avg_time_ms << " ms"
                << ", bandwidth=" << bandwidth_gbs << " GB/s"
                << " (effective, input read " << num_bin_passes << "x)"
                << ", errors=" << error_count
                << ", histogram_sum=" << total_hist_sum
                << " (expected ~" << static_cast<int>(N * 0.9f) << ")"
                << "  " << (passed ? "PASS" : "FAIL") << std::endl;

      csv_file << N << "," << length << ","
               << num_bin_passes << ","
               << avg_time_ms << "," << bandwidth_gbs << ","
               << error_count << "," << (passed ? "1" : "0") << std::endl;

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_input);
      cudaFree(d_hist_blocks);
      free(h_input);
      free(h_hist_ref);
      free(h_hist_gpu);
      free(h_blocks);

    } catch (const std::exception& e) {
      std::cerr << "  [ERROR] Out of memory or exception: " << e.what()
                << std::endl;
      out_of_memory = true;

      if (d_input)       cudaFree(d_input);
      if (d_hist_blocks) cudaFree(d_hist_blocks);
      if (h_input)    free(h_input);
      if (h_hist_ref) free(h_hist_ref);
      if (h_hist_gpu) free(h_hist_gpu);
      if (h_blocks)   free(h_blocks);
    } catch (...) {
      std::cerr << "  [ERROR] Unknown exception" << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory) {
      std::cout << "  [OK] Finished N=" << N << ", bins=" << length
                << std::endl;
    } else {
      csv_file << N << "," << length << "," << num_bin_passes
               << ",OOM,OOM,0,0" << std::endl;
    }
  }

  csv_file.close();
  std::cout << "\n========================================" << std::endl;
  std::cout << "Benchmark completed. Results -> "
               "course6/histogram_v3_benchmark.csv" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
