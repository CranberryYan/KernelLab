#include <random>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>

// block: 512
// thread: 512
#define GRID_SIZE 512
#define BLOCK_SIZE 512
__global__ void histogram_v2(int* input, int* hist, int n, 
                              int low, int high, int length) {
  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  int gid = blockIdx.x * blockDim.x + tid;

  extern __shared__ int s_hist[];

  // 初始化 shared histogram
  for (int i = tid; i < length; i += num_threads) {
    s_hist[i] = 0;
  }
  __syncthreads();

  // 处理数据
  int stride = gridDim.x * blockDim.x;
  for (int i = gid; i < n; i += stride) {
    int val = input[i];
    if (val >= low && val < high) {
      atomicAdd(&s_hist[val - low], 1);
    }
  }
  __syncthreads();

  // 最终 reduce 到 global (只在有值的 bin 上做)
  for (int i = tid; i < length; i += num_threads) {
    if (s_hist[i] > 0) {
      atomicAdd(&hist[i], s_hist[i]);
    }
  }
}

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

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

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
  };

  const int low = -128; // 固定下限
  const int warmup_time = 10;
  const int repeat_time = 100;

  const int grid_size = 512;
  const int block_size = 512;

  // ---------- CSV 文件 ----------
  std::ofstream csv_file("./course6/histogram_v2_benchmark.csv");
  csv_file << "N,Length,GPU_Time_ms,Bandwidth_GBs,ErrorCount,Pass" << std::endl;

  std::cout << "========================================" << std::endl;
  std::cout << "histogram_v2 Benchmark" << std::endl;
  std::cout << "========================================" << std::endl;

  for (const auto& cfg : configs) {
    int N = cfg.n;
    int length = cfg.length;
    int high = low + length;

    std::cout << "\n--- Testing: N=" << N << ", bins=" << length << " ---"
              << std::endl;

    size_t input_bytes = N * sizeof(int);
    size_t hist_bytes = length * sizeof(int);
    int total_hist_sum = 0;

    // ---------- Host 内存 ----------
    int* h_input = nullptr;
    int* h_hist_ref = nullptr;
    int* h_hist_gpu = nullptr;

    // ---------- Device 内存 ----------
    int *d_input = nullptr;
    int *d_hist = nullptr;

    bool out_of_memory = false;

    try {
      h_input = (int*)malloc(input_bytes);
      h_hist_ref = (int*)malloc(hist_bytes);
      h_hist_gpu = (int*)malloc(hist_bytes);
      if (!h_input || !h_hist_ref || !h_hist_gpu) {
        throw std::bad_alloc();
      }

      checkCudaError(cudaMalloc(&d_input, input_bytes),
                     "cudaMalloc d_input failed");
      checkCudaError(cudaMalloc(&d_hist, hist_bytes),
                     "cudaMalloc d_hist failed");

      // ---------- 数据初始化 (固定种子, 可复现) ----------
      std::mt19937 rng(12345);
      std::uniform_int_distribution<int> dist(low, high - 1);
      // 同时混入 10% 的越界值，测试边界处理是否正确
      std::uniform_int_distribution<int> dist_oob(low - 500, high + 500);
      std::uniform_real_distribution<float> dist_oob_ratio(0.0f, 1.0f);

      for (int i = 0; i < N; ++i) {
        if (dist_oob_ratio(rng) < 0.1f) {
          h_input[i] = dist_oob(rng); // 越界值, 不应被计入 histogram_v2
        } else {
          h_input[i] = dist(rng);
        }
      }

      cpu_histogram(h_input, h_hist_ref, N, low, high);

      checkCudaError(cudaMemcpy(d_input, h_input, input_bytes,
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy d_input failed");
      checkCudaError(cudaMemset(d_hist, 0, hist_bytes),
                     "cudaMemset d_hist failed");

      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start),
                     "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop),
                     "cudaEventCreate(stop) failed");

      for (int w = 0; w < warmup_time; ++w) {
        checkCudaError(cudaMemset(d_hist, 0, hist_bytes),
                       "cudaMemset(warmup) failed");
        histogram_v2<<<grid_size, block_size, length * sizeof(int)>>>(
            d_input, d_hist, N, low, high, length);
      }
      cudaDeviceSynchronize();
      checkCudaError(cudaGetLastError(), "Warmup kernel failed");

      float total_time_ms = 0.0f;
      for (int r = 0; r < repeat_time; ++r) {
        checkCudaError(cudaMemset(d_hist, 0, hist_bytes),
                       "cudaMemset(repeat) failed");
        checkCudaError(cudaEventRecord(start),
                       "cudaEventRecord(start) failed");
        histogram_v2<<<grid_size, block_size, length * sizeof(int)>>>(
            d_input, d_hist, N, low, high, length);
        checkCudaError(cudaEventRecord(stop),
                       "cudaEventRecord(stop) failed");
        checkCudaError(cudaEventSynchronize(stop),
                       "cudaEventSynchronize(stop) failed");
        float elapsed_ms = 0.0f;
        checkCudaError(cudaEventElapsedTime(&elapsed_ms, start, stop),
                       "cudaEventElapsedTime failed");
        total_time_ms += elapsed_ms;
      }
      float avg_time_ms = total_time_ms / repeat_time;

      checkCudaError(cudaMemcpy(h_hist_gpu, d_hist, hist_bytes,
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy d_hist to host failed");

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

      float bandwidth_gbs = (input_bytes / (avg_time_ms * 1e6f));
      std::cout << "  [Result] avg_time=" << avg_time_ms << " ms"
                << ", bandwidth=" << bandwidth_gbs << " GB/s"
                << ", errors=" << error_count
                << ", histogram_sum=" << total_hist_sum
                << " (expected ~" << static_cast<int>(N * 0.9f) << ")"
                << "  " << (passed ? "PASS" : "FAIL") << std::endl;

      csv_file << N << "," << length << ","
               << avg_time_ms << "," << bandwidth_gbs << ","
               << error_count << "," << (passed ? "1" : "0") << std::endl;

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_input);
      cudaFree(d_hist);
      free(h_input);
      free(h_hist_ref);
      free(h_hist_gpu);

    } catch (const std::exception& e) {
      std::cerr << "  [ERROR] Out of memory or exception: " << e.what()
                << std::endl;
      out_of_memory = true;

      if (d_input) cudaFree(d_input);
      if (d_hist) cudaFree(d_hist);
      if (h_input) free(h_input);
      if (h_hist_ref) free(h_hist_ref);
      if (h_hist_gpu) free(h_hist_gpu);
    } catch (...) {
      std::cerr << "  [ERROR] Unknown exception" << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory) {
      std::cout << "  [OK] Finished N=" << N << ", bins=" << length
                << std::endl;
    } else {
      csv_file << N << "," << length << ",OOM,OOM,0,0" << std::endl;
    }
  }

  csv_file.close();
  std::cout << "\n========================================" << std::endl;
  std::cout << "Benchmark completed. Results -> "
               "course6/histogram_v2_benchmark.csv" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
