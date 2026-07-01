#include <random>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>

#define BIN_GROUP 4096   // 每轮处理的 bin 数 (shared mem = 16KB)
#define GRID_SIZE 512
#define BLOCK_SIZE 512

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void histogram_v4(int* input, int* hist_blocks, int n,
                              int low, int high, int length) {
  extern __shared__ int s_hist[];

  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int stride = gridDim.x * blockDim.x;

  for (int bin_start = 0; bin_start < length; bin_start += BIN_GROUP) {
    int bin_end = min(bin_start + BIN_GROUP, length);
    int cur_bins = bin_end - bin_start;

    for (int i = tid; i < cur_bins; i += num_threads) {
      s_hist[i] = 0;
    }
    __syncthreads();

    for (int i = gid; i < n; i += stride) {
      int val = input[i];
      int bin = val - low;
      if (bin >= bin_start && bin < bin_end) {
        atomicAdd(&s_hist[bin - bin_start], 1);   // shared atomic only
      }
    }
    __syncthreads();

    for (int i = tid; i < cur_bins; i += num_threads) {
      hist_blocks[blockIdx.x * length + bin_start + i] = s_hist[i];
    }
    __syncthreads();  // 确保写完再进入下一轮 bin group
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

int main() {
  struct TestConfig {
    int n;
    int length;
  };

  std::vector<TestConfig> configs = {
    {       16384,     256 },
    {      131072,     256 },
    {     1048576,     256 },
    {     4194304,     256 },
    {     131072*128,     512 },
    {     131072*128,    1024 },
    {     131072*128,    2048 },
    {     131072*128,    4096 },
    {     524288*128,   16384 },
    {     524288*128,   65536 },
  };

  const int low = -128;
  const int warmup_time = 10;
  const int repeat_time = 1024;

  const int grid_size = GRID_SIZE;
  const int block_size = BLOCK_SIZE;
  const size_t smem_bytes = BIN_GROUP * sizeof(int);  // 固定分配, 每轮复用

  // ---------- CSV ----------
  std::ofstream csv_file("./course6/histogram_v4_benchmark.csv");
  csv_file << "N,Length,BinPasses,GPU_Time_ms,Bandwidth_GBs,ErrorCount,Pass"
           << std::endl;

  std::cout << "========================================" << std::endl;
  std::cout << "histogram_v4 (Single-Kernel Split-by-Length) Benchmark"
            << std::endl;
  std::cout << "========================================" << std::endl;

  for (const auto& cfg : configs) {
    int N = cfg.n;
    int length = cfg.length;
    int high = low + length;
    int num_bin_passes = CEIL_DIV(length, BIN_GROUP);

    std::cout << "\n--- Testing: N=" << N << ", bins=" << length
              << " (" << num_bin_passes << " pass(es)) ---" << std::endl;

    size_t input_bytes   = N * sizeof(int);
    size_t hist_bytes    = length * sizeof(int);
    // d_hist_blocks: [length * GRID_SIZE] — 所有 bin group 的 per-block 结果
    size_t blocks_bytes  = (size_t)length * GRID_SIZE * sizeof(int);
    int total_hist_sum = 0;

    int* h_input     = nullptr;
    int* h_hist_ref  = nullptr;
    int* h_hist_gpu  = nullptr;
    int* h_blocks    = nullptr;

    int *d_input       = nullptr;
    int *d_hist_blocks = nullptr;

    bool out_of_memory = false;

    try {
      h_input    = (int*)malloc(input_bytes);
      h_hist_ref = (int*)malloc(hist_bytes);
      h_hist_gpu = (int*)malloc(hist_bytes);
      h_blocks   = (int*)malloc(blocks_bytes);
      if (!h_input || !h_hist_ref || !h_hist_gpu || !h_blocks) {
        throw std::bad_alloc();
      }

      checkCudaError(cudaMalloc(&d_input, input_bytes),
                     "cudaMalloc d_input failed");
      checkCudaError(cudaMalloc(&d_hist_blocks, blocks_bytes),
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

      cpu_histogram(h_input, h_hist_ref, N, low, high);

      checkCudaError(cudaMemcpy(d_input, h_input, input_bytes,
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy d_input failed");

      // ---------- CUDA Events ----------
      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start)");
      checkCudaError(cudaEventCreate(&stop),  "cudaEventCreate(stop)");

      // ---------- Warmup (单次 kernel launch!) ----------
      for (int w = 0; w < warmup_time; ++w) {
        histogram_v4<<<grid_size, block_size, smem_bytes>>>(
            d_input, d_hist_blocks, N, low, high, length);
      }
      cudaDeviceSynchronize();
      checkCudaError(cudaGetLastError(), "Warmup kernel failed");

      // ---------- 计时 (单次 kernel launch!) ----------
      float total_time_ms = 0.0f;
      for (int r = 0; r < repeat_time; ++r) {
        checkCudaError(cudaEventRecord(start), "cudaEventRecord(start)");
        histogram_v4<<<grid_size, block_size, smem_bytes>>>(
            d_input, d_hist_blocks, N, low, high, length);
        checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop)");
        checkCudaError(cudaEventSynchronize(stop),
                       "cudaEventSynchronize(stop)");
        float elapsed_ms = 0.0f;
        checkCudaError(cudaEventElapsedTime(&elapsed_ms, start, stop),
                       "cudaEventElapsedTime");
        total_time_ms += elapsed_ms;
      }
      float avg_time_ms = total_time_ms / repeat_time;

      // ---------- CPU reduce: sum across blocks ----------
      // 再跑一次取最终结果
      histogram_v4<<<grid_size, block_size, smem_bytes>>>(
          d_input, d_hist_blocks, N, low, high, length);
      cudaDeviceSynchronize();
      checkCudaError(cudaGetLastError(), "Final kernel failed");

      checkCudaError(cudaMemcpy(h_blocks, d_hist_blocks, blocks_bytes,
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy d_hist_blocks failed");

      memset(h_hist_gpu, 0, hist_bytes);
      for (int bin = 0; bin < length; ++bin) {
        int sum = 0;
        for (int bid = 0; bid < GRID_SIZE; ++bid) {
          sum += h_blocks[bid * length + bin];
        }
        h_hist_gpu[bin] = sum;
      }

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
      std::cerr << "  [ERROR] " << e.what() << std::endl;
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
               "course6/histogram_v4_benchmark.csv" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
