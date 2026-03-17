#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

template <typename T>
void prefix_sum_CPU(T* in, T* out, int32_t len) {
  for (int i = 0; i < len; ++i) {
    out[i] = in[i];
  }
  for (int i = 1; i < len; ++i) {
    out[i] = out[i - 1] + out[i];
  }
}

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      printf("CUDA error at %s:%d -> %s\n", __FILE__, __LINE__,               \
             cudaGetErrorString(err__));                                      \
      return false;                                                           \
    }                                                                         \
  } while (0)

struct BenchmarkCase {
  int n;
  int iters;
  int warmup;
  const char* name;
};

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void prefix_sum_inclusive_dlb_optimized(
    T* g_out, const T* g_in, int n,
    T* g_tile_agg, T* g_tile_prefix, int* g_state, int epoch);

__device__ __forceinline__ int smem_padded_idx(int idx) {
  return idx + (idx >> 5);  // one pad every 32 ints
}

static bool verify_result_sampled(const int* out, const int* ref, int n) {
  if (n <= 0) {
    return true;
  }
  const int fixed_idx[] = {0, 1, 2, 3, 7, 15, 31, 63};
  for (int idx : fixed_idx) {
    if (idx < n && out[idx] != ref[idx]) {
      printf("verify failed @%d: out=%d ref=%d\n", idx, out[idx], ref[idx]);
      return false;
    }
  }
  for (int k = 1; k <= 8; ++k) {
    int idx = (int)(((int64_t)n * k) / 9);
    if (idx >= n) {
      idx = n - 1;
    }
    if (out[idx] != ref[idx]) {
      printf("verify failed @%d: out=%d ref=%d\n", idx, out[idx], ref[idx]);
      return false;
    }
  }
  if (out[n - 1] != ref[n - 1]) {
    printf("verify failed @%d: out=%d ref=%d\n", n - 1, out[n - 1], ref[n - 1]);
    return false;
  }
  return true;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
static bool run_benchmark_case(const BenchmarkCase& bc) {
  constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;
  const int numTiles = (bc.n + TILE_ITEMS - 1) / TILE_ITEMS;
  const int32_t size = bc.n * (int)sizeof(int);

  int* h_in = reinterpret_cast<int*>(malloc(size));
  int* h_out = reinterpret_cast<int*>(malloc(size));
  int* h_ref = reinterpret_cast<int*>(malloc(size));
  if (!h_in || !h_out || !h_ref) {
    printf("host malloc failed\n");
    free(h_in);
    free(h_out);
    free(h_ref);
    return false;
  }

  for (int i = 0; i < bc.n; ++i) {
    h_in[i] = i % 10;
  }
  prefix_sum_CPU<int>(h_in, h_ref, bc.n);

  int *d_in, *d_out, *d_state;
  int *d_tile_agg, *d_tile_prefix;
  CUDA_CHECK(cudaMalloc(&d_in, size));
  CUDA_CHECK(cudaMalloc(&d_out, size));
  CUDA_CHECK(cudaMalloc(&d_tile_agg, numTiles * (int)sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_tile_prefix, numTiles * (int)sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_state, numTiles * (int)sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_state, 0, numTiles * (int)sizeof(int)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int w = 0; w < bc.warmup; ++w) {
    prefix_sum_inclusive_dlb_optimized<int, BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<dim3(numTiles), dim3(BLOCK_THREADS)>>>(d_out, d_in, bc.n,
                                                  d_tile_agg, d_tile_prefix,
                                                  d_state, w + 1);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start, 0));
  for (int i = 0; i < bc.iters; ++i) {
    prefix_sum_inclusive_dlb_optimized<int, BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<dim3(numTiles), dim3(BLOCK_THREADS)>>>(d_out, d_in, bc.n,
                                                  d_tile_agg, d_tile_prefix,
                                                  d_state, i + 1000);
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / bc.iters;
  const double elems_per_s = (double)bc.n / (avg_ms * 1e-3);
  const double gib_per_s =
      ((double)bc.n * sizeof(int) * 2.0) / (avg_ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);

  CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
  const bool ok = verify_result_sampled(h_out, h_ref, bc.n);

  printf("[%-12s] n=%-10d tiles=%-7d avg=%.6f ms  throughput=%.3f Gelem/s  "
         "est_bw=%.3f GiB/s  %s\n",
         bc.name, bc.n, numTiles, avg_ms, elems_per_s / 1e9, gib_per_s,
         ok ? "OK" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_tile_agg));
  CUDA_CHECK(cudaFree(d_tile_prefix));
  CUDA_CHECK(cudaFree(d_state));
  free(h_in);
  free(h_out);
  free(h_ref);
  return ok;
}

// 单kernel + DLB(Decoupled Look-Back)的inclusive scan
// 1. 每个block先独立算出自己tile的块内prefix_sum
// 2. 全局状态机(g_state/g_tile_agg/g_tile_prefix)解析前面所有tile的和
// 3. 最后再把这个carry加到本tile的局部prefix_sum, 写到g_out
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void prefix_sum_inclusive_dlb_optimized(
    T* g_out, const T* g_in, int n,
    T* g_tile_agg, T* g_tile_prefix, int* g_state, int epoch) {
  // thread layout
  //  constexpr int BLOCK_THREADS = 256;
  //  constexpr int ITEMS_PER_THREAD = 4;
  //  constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;
  //  一个tile = 1024个元素, 由一个block处理
  static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be warp-aligned");
  constexpr int WARP_SIZE = 32;
  constexpr int WARP_COUNT = BLOCK_THREADS / WARP_SIZE;
  constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int SMEM_PAD = (TILE_ITEMS + WARP_SIZE - 1) / WARP_SIZE;

  const int tid = threadIdx.x;
  const int lane = tid & (WARP_SIZE - 1);
  const int warp = tid / WARP_SIZE;
  const int tile = blockIdx.x;
  const int tile_base = tile * TILE_ITEMS;

  __shared__ T warp_inclusive[WARP_COUNT];
  __shared__ T s_tile_sum;
  __shared__ T s_tile_exclusive;
  __shared__ T s_exchange[TILE_ITEMS + SMEM_PAD];

  // 每个thread的local var
  T vals[ITEMS_PER_THREAD];
  T local_prefix[ITEMS_PER_THREAD];

  // 数据流, global -> shared(striped, coalesced) -> reg(blocked, 原算法顺序)
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    // 假设i = 0, tile_base = 0
    //  tid0:   idx: 0    addr: 0x0000
    //  tid1:   idx: 1    addr: 0x0004
    //  ...
    //  tid31:  idx: 31   addr: 0x007C
    //  ...
  
    // i = 1
    //  tid0:   idx: 256
    //  tid1:   idx: 257
    //  ...
    //  tid31:  idx: 287
    // 一次循环, 需要load 32 * 4 = 128B, 地址集中在0x0000~0x007C, 覆盖128B的窗口
    // 一次warp load指令总共只请求128B有效数据, 需要load 1次才行
    int striped_idx = tile_base + i * BLOCK_THREADS + tid;

    // 但是此时的s_exchange中的元素排列, 不是按照idx: 0 ~ 3对应tid0要处理的元素
    //  所以不能直接load到var中

    // 优点: 合并访存
    // 缺点:
    //  原: global -> reg
    //  现: global -> smem -> reg
    // 如果global -> reg可以做到访存合并, 那么就不需要经过smem, 但是本kernel的情况下
    // stride为4, 很差, 所以需要经过smem中转
    // s_exchange[i * BLOCK_THREADS + tid] = (striped_idx < n) ?
    //                                         g_in[striped_idx] :
    //                                         T(0);
  
    s_exchange[smem_padded_idx(i * BLOCK_THREADS + tid)] = (striped_idx < n) ?
                                                      g_in[striped_idx] :
                                                      T(0);
  }
  __syncthreads();

  // 重新计算idx, load到reg中
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    // bank conflict
    // i = 0
    //  tid0: 0
    //  tid1: 4
    //  ...
    //  tid31: 124
    // tid0 tid8 tid16 tid24会访问同一个bank
    // vals[i] = s_exchange[tid * ITEMS_PER_THREAD + i];
    vals[i] = s_exchange[smem_padded_idx(tid * ITEMS_PER_THREAD + i)];
  }

  // 每个thread先做自己的4个元素的prefix_sum
  T thread_sum = T(0);
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    thread_sum += vals[i];
    local_prefix[i] = thread_sum;
  }

  // shuffle
  // __shfl_up_sync
  // 数据向上平移
  // 在一个warp中, lane_id为t的thread将从lane_id为t - delta的thread中读取val的值,
  //  若t - delta < 0, 则保留自身的val
  // 0 1 2 3 ... 31
  // A B C D ... Z
  // A B A B ... X(delta为2, AB保留(t - delta < 0))

  // warp内用shfl做线程和的前缀
  // 假设WARP_SIZE: 8
  // lane:        0 1 2 3 4 5 6 7
  // thread_sum:  1 2 3 4 5 6 7 8
  // offset: 1
  // nbor:        1 1 2 3 4 5 6 7
  // lane >= 1
  // warp_scan:   1 3 5 7 9 11 13 15
  // offset: 2
  // nbor:        1 3 1 3 5  7  9 11
  // lane >= 2
  // warp_scan:   1 3 6 10 14 18 22 26
  // offset: 4
  // nbor:        1 3 6 10  1  3  6 10
  // lane >= 4
  // warp_scan:   1 3 6 10 15 21 28 36
  // tid: WARP_SIZE - 1, 保存着对应warp中的WARP_SIZE * 4个元素的agg
  T warp_scan = thread_sum;
  #pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    T nbor = __shfl_up_sync(0xffffffff, warp_scan, offset);
    if (lane >= offset) {
      warp_scan += nbor;
    }
  }

  // warp_inclusive: 每个warp的agg
  // 接下来对该数组求prefix
  //  使得每个warp拥有前面warp的信息, 且最后一个元素是该block的agg
  if (lane == WARP_SIZE - 1) {
    warp_inclusive[warp] = warp_scan;
  }
  __syncthreads();

  if (warp == 0) {
    // lane < WARP_COUNT: 隐含条件, WARP_COUNT <= 32(thradDim < 1024)
    T x = (lane < WARP_COUNT) ? warp_inclusive[lane] : T(0);
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
      T nbor = __shfl_up_sync(0xffffffff, x, offset);
      if (lane >= offset) {
        x += nbor;
      }
    }
    if (lane < WARP_COUNT) {
      warp_inclusive[lane] = x;
    }
  }
  __syncthreads();

  // 当前warp之前(ex)所有warp的总和是多少
  T warp_exclusive = (warp == 0) ? T(0) : warp_inclusive[warp - 1];

  // warp_scan:         1 3 6 10 15 21 28 36
  // thread_sum:        1 2 3  4  5  6  7  8
  // thread_exclusive:  0 1 3  6 10 15 21 28
  // warp_scan: 是inclusive, 包含了当前thread自己的thread_sum
  // warp_scan - thread_sum: 变成了thread_exclusive, 但是没有其他warp的信息
  // warp_exclusive + (warp_scan - thread_sum): 加入了warp信息
  // thread_exclusive: 在当前block中的每个thread的exclusive
  T thread_exclusive = warp_exclusive + (warp_scan - thread_sum);

  // local_prefix: 只是thread内部4个元素的prefix_sum, 还没包含
  //  前面所有warp的信息, 当前warp前面所有thread的信息
  // local_prefix[i] += thread_exclusive;
  //  local_prefix[i] -> thread_inclusive
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    local_prefix[i] += thread_exclusive;
  }

  // s_tile_sum: 当前tile(block)中的agg, 最后一个thread的最后一个元素
  //  隐含条件, 每个tile(block)是满元素的
  if (tid == BLOCK_THREADS - 1) {
    s_tile_sum = local_prefix[ITEMS_PER_THREAD - 1];
  }
  __syncthreads();

  // 不同tile/block之间已经算出来局部和, 需要和其他tile/block信息交互, 拿到offset
  if (tid == 0) {
    // epoch:
    //  说明g_state不是简答用: 0, 1, 2
    const int agg_ready = epoch * 2 + 1;
    const int pref_ready = epoch * 2 + 2;

    // 先把当前tile的总和写到global
    //  第tile/block的局部总和, 该tile/block的agg
    g_tile_agg[tile] = s_tile_sum;
    __threadfence();

    // 第tile/block的状态更新, agg可用
    g_state[tile] = agg_ready;

    // 开始look-back
    T carry = T(0);
    int look = tile - 1;

    // ??? 不会有时序问题吗 ???
    volatile int* v_state = reinterpret_cast<volatile int*>(g_state);
    while (look >= 0) {
      int s = 0;
      do {
        // 先至少等chunki-1把agg发布出来, 否则一直等
        s = v_state[look];
      } while (s < agg_ready);

      // 如果chunki-1连prefix发布了, 那就不看了, 直接拿到prefixi-1
      // 否则拿到chunki-1的agg, 然后继续看i-2

      // offseti = prefixi-1
      // offseti = prefixi-2 + aggi-1
      // offseti = prefixi-3 + aggi-2 + aggi-1
      if (s >= pref_ready) {
        carry += g_tile_prefix[look];
        break;
      } else {
        carry += g_tile_agg[look];
        --look;
      }
    }
    // ??? 不会有时序问题吗 ???

    // s_tile_exclusive: 当前tile/block的全局offset
    // 可能是
    //  prefixi-1
    //  aggregate0 + ... + aggregatei-1
    //  prefix2 + aggregate3 + ... + aggregatei-1
    s_tile_exclusive = carry;

    // 第tile/block的inclusive
    // 当前tile的tile级inclusive prefix, 也就是从第0个tile到当前tile的aggregate总和
    g_tile_prefix[tile] = carry + s_tile_sum;
    __threadfence();
    g_state[tile] = pref_ready;
  }
  __syncthreads();

  // 写回: reg(blocked) -> shared -> global(striped, coalesced)
  T add = s_tile_exclusive;
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    // s_exchange[tid * ITEMS_PER_THREAD + i] = local_prefix[i] + add;
    s_exchange[smem_padded_idx(tid * ITEMS_PER_THREAD + i)] = local_prefix[i] + add;
  }
  __syncthreads();

  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int striped_idx = tile_base + i * BLOCK_THREADS + tid;
    if (striped_idx < n) {
      // g_out[striped_idx] = s_exchange[i * BLOCK_THREADS + tid];
      g_out[striped_idx] = s_exchange[smem_padded_idx(i * BLOCK_THREADS + tid)];
    }
  }
}

int main() {
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = 4;
  constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

  printf("v4.5 benchmark (single-kernel DLB, coalesced IO)\n");
  printf("config: block=%d, items/thread=%d, tile=%d\n",
         BLOCK_THREADS, ITEMS_PER_THREAD, TILE_ITEMS);

  const BenchmarkCase cases[] = {
      {1024 * 64,   20000, 20, "small-64k"},
      {1024 * 800,  10000, 10, "mid-800k"},
      {1024 * 1024, 5000,  10, "max-1m"},
      {1024 * 4096, 2000,  10, "large-4m"},
  };

  bool all_ok = true;
  for (const auto& bc : cases) {
    bool ok = run_benchmark_case<BLOCK_THREADS, ITEMS_PER_THREAD>(bc);
    all_ok = all_ok && ok;
  }

  printf("benchmark done: %s\n", all_ok ? "ALL PASS" : "HAS FAILURES");
  return all_ok ? 0 : 1;
}
