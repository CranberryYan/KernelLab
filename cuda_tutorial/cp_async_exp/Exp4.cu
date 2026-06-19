// 实验4: producer-consumer double buffer pipeline mini

// compute_iters = 0     | check no_pipe/db = PASS/PASS/PASS
//     sync                      :   0.4704 ms, input-read BW =   570.71 GB/s
//     async no pipeline         :   0.4869 ms, input-read BW =   551.30 GB/s, speedup vs sync = 0.966x
//     async double buf          :   0.3073 ms, input-read BW =   873.57 GB/s, speedup vs no_pipe = 1.585x, speedup vs sync = 1.531x
//     async double buf no pipe  :   0.4422 ms, input-read BW =   607.07 GB/s, speedup vs no_pipe = 1.101x, speedup vs sync = 1.064x

// compute_iters = 4     | check no_pipe/db = PASS/PASS/PASS
//     sync                      :   0.5018 ms, input-read BW =   534.91 GB/s
//     async no pipeline         :   0.5186 ms, input-read BW =   517.65 GB/s, speedup vs sync = 0.968x
//     async double buf          :   0.3121 ms, input-read BW =   860.04 GB/s, speedup vs no_pipe = 1.661x, speedup vs sync = 1.608x
//     async double buf no pipe  :   0.4774 ms, input-read BW =   562.29 GB/s, speedup vs no_pipe = 1.086x, speedup vs sync = 1.051x

// compute_iters = 16    | check no_pipe/db = PASS/PASS/PASS
//     sync                      :   0.6016 ms, input-read BW =   446.18 GB/s
//     async no pipeline         :   0.6188 ms, input-read BW =   433.80 GB/s, speedup vs sync = 0.972x
//     async double buf          :   0.3370 ms, input-read BW =   796.61 GB/s, speedup vs no_pipe = 1.836x, speedup vs sync = 1.785x
//     async double buf no pipe  :   0.5788 ms, input-read BW =   463.77 GB/s, speedup vs no_pipe = 1.069x, speedup vs sync = 1.039x

// compute_iters = 64    | check no_pipe/db = PASS/PASS/PASS
//     sync                      :   0.9882 ms, input-read BW =   271.63 GB/s
//     async no pipeline         :   1.0133 ms, input-read BW =   264.92 GB/s, speedup vs sync = 0.975x
//     async double buf          :   0.6187 ms, input-read BW =   433.84 GB/s, speedup vs no_pipe = 1.638x, speedup vs sync = 1.597x
//     async double buf no pipe  :   0.9783 ms, input-read BW =   274.38 GB/s, speedup vs no_pipe = 1.036x, speedup vs sync = 1.010x

// compute_iters = 256   | check no_pipe/db = PASS/PASS/PASS
//     sync                      :   2.5865 ms, input-read BW =   103.78 GB/s
//     async no pipeline         :   2.5909 ms, input-read BW =   103.61 GB/s, speedup vs sync = 0.998x
//     async double buf          :   2.1887 ms, input-read BW =   122.65 GB/s, speedup vs no_pipe = 1.184x, speedup vs sync = 1.182x
//     async double buf no pipe  :   2.6381 ms, input-read BW =   101.75 GB/s, speedup vs no_pipe = 0.982x, speedup vs sync = 0.980x

#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(call)                                                \
  do {                                                                  \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
              __FILE__, __LINE__, cudaGetErrorString(err));             \
      std::exit(EXIT_FAILURE);                                          \
    }                                                                   \
  } while (0)

__device__ __forceinline__ float compute_vec(float4 v, int compute_iters) {
  float x = v.x + v.y * 0.5f + v.z * 0.25f + v.w * 0.125f;

  // 人为增加计算量，用来观察 copy/compute overlap。
  // 不做 unroll，方便通过 runtime 参数调节计算量。
  #pragma unroll 1
  for (int i = 0; i < compute_iters; ++i) {
      x = fmaf(x, 1.000001f, 0.000001f);
  }

  return x;
}

template<int BLOCK_SIZE, int TILES_PER_BLOCK>
__global__ void sync_copy_compute_kernel(const float4* __restrict__ in,
                                         float* __restrict__ out,
                                         int compute_iters) {
  int tid = threadIdx.x;
  int out_gid = blockIdx.x * BLOCK_SIZE + tid;
  int block_tile_base = blockIdx.x * TILES_PER_BLOCK * BLOCK_SIZE;

  __shared__ float4 smem[BLOCK_SIZE];

  float acc = 0.0f;

  #pragma unroll 1
  for (int tile = 0; tile < TILES_PER_BLOCK; ++tile) {
    int gid = block_tile_base + tile * BLOCK_SIZE + tid;

    smem[tid]= in[gid];
    __syncthreads();

    float4 s = smem[tid];
    acc += compute_vec(s, compute_iters);
    __syncthreads();
  }

  out[out_gid] = acc;
}

template<int BLOCK_SIZE, int TILES_PER_BLOCK>
__global__ void async_no_pipeline_kernel(const float4* __restrict__ in,
                                          float* __restrict__ out,
                                          int compute_iters) {
  int tid = threadIdx.x;
  int out_gid = blockIdx.x * BLOCK_SIZE + tid;
  int block_tile_base = blockIdx.x * TILES_PER_BLOCK * BLOCK_SIZE;

  __shared__ float4 smem[BLOCK_SIZE];

  float acc = 0.0f;

  #pragma unroll 1
  for (int tile = 0; tile < TILES_PER_BLOCK; ++tile) {
    int gid = block_tile_base + tile * BLOCK_SIZE + tid;

    __pipeline_memcpy_async(&smem[tid], &in[gid], sizeof(float4));
    __pipeline_commit();

    // 没有重叠, commit后立刻wait
    __pipeline_wait_prior(0);
    __syncthreads();

    float4 s = smem[tid];
    acc += compute_vec(s, compute_iters);
    __syncthreads();
  }

  out[out_gid] = acc;
}

template<int BLOCK_SIZE, int TILES_PER_BLOCK>
__global__ void async_double_buffer_kernel(const float4* __restrict__ in,
                                           float* __restrict__ out,
                                           int compute_iters) {
  int tid = threadIdx.x;
  int out_gid = blockIdx.x * BLOCK_SIZE + tid;
  int block_tile_base = blockIdx.x * TILES_PER_BLOCK * BLOCK_SIZE;

  __shared__ float4 smem[2][BLOCK_SIZE];

  float acc = 0.0f;

  // prologue
  // tile0
  int gid0 = block_tile_base + 0 * BLOCK_SIZE + tid;
  __pipeline_memcpy_async(&smem[0][tid], &in[gid0], sizeof(float4));
  __pipeline_commit();

  // tile1
  if (TILES_PER_BLOCK > 1) {
    int gid1 = block_tile_base + 1 * BLOCK_SIZE + tid;
    __pipeline_memcpy_async(&smem[1][tid], &in[gid1], sizeof(float4));
    __pipeline_commit();
  }

  // main loop
  int PingPong = 0;

  #pragma unroll 1
  for (int tile = 0; tile < TILES_PER_BLOCK - 1; ++tile) {
    // 允许最近的 1 个 copy group 还在触发
    // 更早的必须完成
    __pipeline_wait_prior(1);
    __syncthreads();

    float4 s = smem[PingPong][tid];
    acc += compute_vec(s, compute_iters);
    __syncthreads(); // 为了模拟真实的gemm

    int next_tile = tile + 2;
    if (next_tile < TILES_PER_BLOCK) {
      int gid_next = block_tile_base + next_tile * BLOCK_SIZE + tid;
      __pipeline_memcpy_async(&smem[PingPong][tid], &in[gid_next], sizeof(float4));
      __pipeline_commit();
    }

    PingPong = 1 - PingPong;
  }

  // tail
  __pipeline_wait_prior(0);
  __syncthreads();

  float4 s = smem[PingPong][tid];
  acc += compute_vec(s, compute_iters);
  __syncthreads();

  out[out_gid] = acc;
}

template<int BLOCK_SIZE, int TILES_PER_BLOCK>
__global__ void sync_double_buffer_no_pipeline_kernel(
    const float4* __restrict__ in,
    float* __restrict__ out,
    int compute_iters) {
  int tid = threadIdx.x;
  int out_gid = blockIdx.x * BLOCK_SIZE + tid;
  int block_tile_base = blockIdx.x * TILES_PER_BLOCK * BLOCK_SIZE;

  __shared__ float4 smem[2][BLOCK_SIZE];

  float acc = 0.0f;

  // prologue
  // tile0
  int gid0 = block_tile_base + 0 * BLOCK_SIZE + tid;
  smem[0][tid] = in[gid0];

  // tile1
  if (TILES_PER_BLOCK > 1) {
    int gid1 = block_tile_base + 1 * BLOCK_SIZE + tid;
    smem[1][tid] = in[gid1];
  }

  // main loop
  int PingPong = 0;

  #pragma unroll 1
  for (int tile = 0; tile < TILES_PER_BLOCK - 1; ++tile) {
    float4 s = smem[PingPong][tid];
    __syncthreads();

    acc += compute_vec(s, compute_iters);

    int next_tile = tile + 2;
    if (next_tile < TILES_PER_BLOCK) {
      int gid_next = block_tile_base + next_tile * BLOCK_SIZE + tid;
      smem[PingPong][tid] = in[gid_next];
    }
    __syncthreads();

    PingPong = 1 - PingPong;
  }

  // tail
  float4 s = smem[PingPong][tid];
  acc += compute_vec(s, compute_iters);

  out[out_gid] = acc;
}

template<int BLOCK_SIZE, int TILES_PER_BLOCK>
void launch_sync(const float4* d_in, float* d_out, int num_blocks, int compute_iters) {
  sync_copy_compute_kernel<BLOCK_SIZE, TILES_PER_BLOCK>
      <<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, compute_iters);
}

template<int BLOCK_SIZE, int TILES_PER_BLOCK>
void launch_async_no_pipeline(const float4* d_in, float* d_out, int num_blocks, int compute_iters) {
  async_no_pipeline_kernel<BLOCK_SIZE, TILES_PER_BLOCK>
      <<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, compute_iters);
}

template<int BLOCK_SIZE, int TILES_PER_BLOCK>
void launch_async_double_buffer(const float4* d_in, float* d_out, int num_blocks, int compute_iters) {
  async_double_buffer_kernel<BLOCK_SIZE, TILES_PER_BLOCK>
      <<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, compute_iters);
}

template<int BLOCK_SIZE, int TILES_PER_BLOCK>
void launch_sync_double_buffer_no_pipeline_kernel(const float4* d_in, float* d_out, int num_blocks, int compute_iters) {
  sync_double_buffer_no_pipeline_kernel<BLOCK_SIZE, TILES_PER_BLOCK>
      <<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, compute_iters);
}


template<typename Launcher>
float benchmark(Launcher launcher, int warmup, int repeat) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
      launcher();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
      launcher();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return ms / repeat;
}

void fill_input(std::vector<float4>& h) {
  for (size_t i = 0; i < h.size(); ++i) {
    float base = static_cast<float>((i % 1009) + 1);
    h[i].x = base * 0.001f;
    h[i].y = base * 0.002f;
    h[i].z = base * 0.003f;
    h[i].w = base * 0.004f;
  }
}

bool check_close(const std::vector<float>& ref,
                 const std::vector<float>& got,
                 const char* name) {
  int mismatches = 0;

  for (size_t i = 0; i < ref.size(); ++i) {
    float a = ref[i];
    float b = got[i];

    float diff = fabsf(a - b);
    float tol = 1e-3f + 1e-5f * fabsf(a);

    if (diff > tol) {
      if (mismatches < 8) {
        printf("%s mismatch at %zu: ref = %.8f, got = %.8f, diff = %.8f\n",
                name, i, a, b, diff);
      }
      mismatches++;
    }
  }

  if (mismatches > 0) {
    printf("%s total mismatches = %d\n", name, mismatches);
    return false;
  }

  return true;
}

template<int BLOCK_SIZE, int TILES_PER_BLOCK>
void run_one_case(int compute_iters,
                  int num_blocks,
                  float4* d_in,
                  float* d_sync,
                  float* d_no_pipe,
                  float* d_db,
                  float* d_db_no_pipe,
                  std::vector<float>& h_sync,
                  std::vector<float>& h_no_pipe,
                  std::vector<float>& h_db,
                  std::vector<float>& h_db_no_pipe,
                  int warmup,
                  int repeat,
                  size_t input_bytes) {
  size_t output_bytes = static_cast<size_t>(num_blocks) * BLOCK_SIZE * sizeof(float);

  CHECK_CUDA(cudaMemset(d_sync, 0, output_bytes));
  CHECK_CUDA(cudaMemset(d_no_pipe, 0, output_bytes));
  CHECK_CUDA(cudaMemset(d_db, 0, output_bytes));
  CHECK_CUDA(cudaMemset(d_db_no_pipe, 0, output_bytes));

  launch_sync<BLOCK_SIZE, TILES_PER_BLOCK>(d_in, d_sync, num_blocks, compute_iters);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  launch_async_no_pipeline<BLOCK_SIZE, TILES_PER_BLOCK>(d_in, d_no_pipe, num_blocks, compute_iters);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  launch_async_double_buffer<BLOCK_SIZE, TILES_PER_BLOCK>(d_in, d_db, num_blocks, compute_iters);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  launch_sync_double_buffer_no_pipeline_kernel<BLOCK_SIZE, TILES_PER_BLOCK>(d_in, d_db_no_pipe, num_blocks, compute_iters);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_sync.data(), d_sync, output_bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_no_pipe.data(), d_no_pipe, output_bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_db.data(), d_db, output_bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_db_no_pipe.data(), d_db_no_pipe, output_bytes, cudaMemcpyDeviceToHost));

  bool no_pipe_ok = check_close(h_sync, h_no_pipe, "async_no_pipeline");
  bool db_ok = check_close(h_sync, h_db, "async_double_buffer");
  bool db_no_pipe_ok = check_close(h_sync, h_db_no_pipe, "async_double_buffer_no_pipeline");

  auto sync_launcher = [&]() {
    launch_sync<BLOCK_SIZE, TILES_PER_BLOCK>(d_in, d_sync, num_blocks, compute_iters);
  };

  auto no_pipe_launcher = [&]() {
    launch_async_no_pipeline<BLOCK_SIZE, TILES_PER_BLOCK>(d_in, d_no_pipe, num_blocks, compute_iters);
  };

  auto db_launcher = [&]() {
    launch_async_double_buffer<BLOCK_SIZE, TILES_PER_BLOCK>(d_in, d_db, num_blocks, compute_iters);
  };

  auto db_no_pipe_launcher = [&]() {
    launch_sync_double_buffer_no_pipeline_kernel<BLOCK_SIZE, TILES_PER_BLOCK>(d_in, d_db_no_pipe, num_blocks, compute_iters);
  };

  float sync_ms = benchmark(sync_launcher, warmup, repeat);
  float no_pipe_ms = benchmark(no_pipe_launcher, warmup, repeat);
  float db_ms = benchmark(db_launcher, warmup, repeat);
  float db_no_pipe_ms = benchmark(db_no_pipe_launcher, warmup, repeat);

  double read_gb = static_cast<double>(input_bytes) / 1e9;

  double sync_bw = read_gb / (sync_ms * 1e-3);
  double no_pipe_bw = read_gb / (no_pipe_ms * 1e-3);
  double db_bw = read_gb / (db_ms * 1e-3);
  double db_no_pipe_bw = read_gb / (db_no_pipe_ms * 1e-3);

  printf("compute_iters = %-5d | check no_pipe/db = %s/%s/%s\n",
          compute_iters,
          no_pipe_ok ? "PASS" : "FAIL",
          db_ok ? "PASS" : "FAIL",
          db_no_pipe_ok ? "PASS" : "FAIL");

  printf("    sync                      : %8.4f ms, input-read BW = %8.2f GB/s\n",
          sync_ms, sync_bw);

  printf("    async no pipeline         : %8.4f ms, input-read BW = %8.2f GB/s, speedup vs sync = %.3fx\n",
          no_pipe_ms, no_pipe_bw, sync_ms / no_pipe_ms);

  printf("    async double buf          : %8.4f ms, input-read BW = %8.2f GB/s, speedup vs no_pipe = %.3fx, speedup vs sync = %.3fx\n",
          db_ms, db_bw, no_pipe_ms / db_ms, sync_ms / db_ms);

  printf("    async double buf no pipe  : %8.4f ms, input-read BW = %8.2f GB/s, speedup vs no_pipe = %.3fx, speedup vs sync = %.3fx\n",
          db_no_pipe_ms, db_no_pipe_bw, no_pipe_ms / db_no_pipe_ms, sync_ms / db_no_pipe_ms);

  printf("\n");
}

int main() {
  constexpr int BLOCK_SIZE = 256;
  constexpr int TILES_PER_BLOCK = 1024;

  // total input payload = 256 MB
  // 每个元素是 float4 = 16B
  size_t input_bytes = 256ull * 1024ull * 1024ull;
  size_t total_vec = input_bytes / sizeof(float4);

  // 保证 total_vec 能整除一个 block 负责的数据量。
  size_t vec_per_cta = static_cast<size_t>(BLOCK_SIZE) * TILES_PER_BLOCK;
  int num_blocks = static_cast<int>(total_vec / vec_per_cta);
  total_vec = static_cast<size_t>(num_blocks) * vec_per_cta;
  input_bytes = total_vec * sizeof(float4);

  size_t output_elems = static_cast<size_t>(num_blocks) * BLOCK_SIZE;
  size_t output_bytes = output_elems * sizeof(float);

  constexpr int WARMUP = 10;
  constexpr int REPEAT = 100;

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  printf("GPU                 = %s\n", prop.name);
  printf("Compute capability  = %d.%d\n", prop.major, prop.minor);
  printf("BLOCK_SIZE          = %d\n", BLOCK_SIZE);
  printf("TILES_PER_BLOCK     = %d\n", TILES_PER_BLOCK);
  printf("num_blocks          = %d\n", num_blocks);
  printf("input payload        = %.2f MB\n", input_bytes / 1024.0 / 1024.0);
  printf("output payload       = %.2f MB\n", output_bytes / 1024.0 / 1024.0);
  printf("copy granularity     = float4, 16B per thread\n\n");

  std::vector<float4> h_in(total_vec);
  std::vector<float> h_sync(output_elems);
  std::vector<float> h_no_pipe(output_elems);
  std::vector<float> h_db(output_elems);
  std::vector<float> h_db_no_pipeline(output_elems);

  fill_input(h_in);

  float4* d_in = nullptr;
  float* d_sync = nullptr;
  float* d_no_pipe = nullptr;
  float* d_db = nullptr;
  float* d_db_no_pipe = nullptr;

  CHECK_CUDA(cudaMalloc(&d_in, input_bytes));
  CHECK_CUDA(cudaMalloc(&d_sync, output_bytes));
  CHECK_CUDA(cudaMalloc(&d_no_pipe, output_bytes));
  CHECK_CUDA(cudaMalloc(&d_db, output_bytes));
  CHECK_CUDA(cudaMalloc(&d_db_no_pipe, output_bytes));

  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), input_bytes, cudaMemcpyHostToDevice));

  printf("==== Exp4: true double buffer cp.async pipeline ====\n\n");

  // 你可以按需增删这些计算量。
  // compute_iters = 0 时，几乎没有计算可重叠，double buffer 收益通常不明显。
  // compute_iters 中等时，最容易观察到 pipeline 的收益。
  int compute_cases[] = {0, 4, 16, 64, 256};

  for (int iters : compute_cases) {
    run_one_case<BLOCK_SIZE, TILES_PER_BLOCK>(
      iters,
      num_blocks,
      d_in,
      d_sync,
      d_no_pipe,
      d_db,
      d_db_no_pipe,
      h_sync,
      h_no_pipe,
      h_db,
      h_db_no_pipeline,
      WARMUP,
      REPEAT,
      input_bytes
    );
  }

  printf("Interpretation:\n");
  printf("1. async_no_pipeline uses cp.async but waits immediately, so it mainly tests cp.async API overhead/path.\n");
  printf("2. async_double_buffer overlaps copy of tile[k+1] with compute of tile[k].\n");
  printf("3. If compute_iters = 0, there is almost no compute to hide memory latency, so double buffer may not win much.\n");
  printf("4. If compute_iters is moderate, double buffer should have a better chance to beat async_no_pipeline.\n");
  printf("5. If compute_iters is very large, memory copy is fully hidden, and extra pipeline benefit may shrink again.\n");

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_sync));
  CHECK_CUDA(cudaFree(d_no_pipe));
  CHECK_CUDA(cudaFree(d_db));
  CHECK_CUDA(cudaFree(d_db_no_pipe));

  return 0;
}
