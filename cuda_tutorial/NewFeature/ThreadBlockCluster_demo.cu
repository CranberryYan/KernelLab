#include <vector>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Helpers for compile-time macro stringification.
#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)


namespace cg = cooperative_groups;

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__)                \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                               	\
    }                                                                        	\
  } while (0)

__global__ void cluster_info_kernel(int* out_cluster_id, int* out_block_rank,
                                    int total_blocks) {
  const int bid = blockIdx.x;
  if (bid >= total_blocks) return;

#if defined(_CG_HAS_CLUSTER_GROUP)
  auto cluster = cg::this_cluster();
  out_block_rank[bid] = static_cast<int>(cluster.block_rank());
  out_cluster_id[bid] = bid / static_cast<int>(cluster.dim_blocks().x);
  cluster.sync();
#else
  out_block_rank[bid] = -1;
  out_cluster_id[bid] = -1;
#endif
}

int main() {
#if CUDART_VERSION < 12000
  std::cout << "Thread Block Cluster demo needs CUDA 12.0+ runtime API.\n";
  return 0;
#else
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name 
            << " (SM " << prop.major << prop.minor << ")\n";

  int cluster_launch_supported = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&cluster_launch_supported,
                                    cudaDevAttrClusterLaunch, 0));
  if (prop.major < 9 || !cluster_launch_supported) {
    std::cout << "not support thread block cluster launch (needs SM90+).\n";
    return 0;
  }

  constexpr int cluster_size_x = 2;  // each cluster has 2 thread blocks
  constexpr int clusters = 4;
  constexpr int blocks = cluster_size_x * clusters;
  constexpr int threads = 128;

  int* d_cluster_id = nullptr;
  int* d_block_rank = nullptr;
  CHECK_CUDA(cudaMalloc(&d_cluster_id, blocks * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_block_rank, blocks * sizeof(int)));

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(blocks, 1, 1);
  config.blockDim = dim3(threads, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = 0;

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster_size_x;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  CHECK_CUDA(cudaLaunchKernelEx(&config, cluster_info_kernel,
                                d_cluster_id, d_block_rank, blocks));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<int> h_cluster_id(blocks), h_block_rank(blocks);
  CHECK_CUDA(cudaMemcpy(h_cluster_id.data(), d_cluster_id,
                        blocks * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_block_rank.data(), d_block_rank,
                        blocks * sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "\nblockIdx.x -> (cluster_id, block_rank_in_cluster)\n";
  for (int b = 0; b < blocks; ++b) {
    std::cout << "block " << b 
              << " -> (" << h_cluster_id[b] << ", " << h_block_rank[b] << ")\n";
  }
  std::cout << "\nExpected pattern for cluster_size_x=2:\n";
  std::cout << "  block ranks repeat as 0,1,0,1,...\n";
  std::cout << "  cluster_id increments every 2 blocks.\n";

  bool compiled_without_cluster_path = true;
  for (int b = 0; b < blocks; ++b) {
    if (h_cluster_id[b] != -1 || h_block_rank[b] != -1) {
      compiled_without_cluster_path = false;
      break;
    }
  }
  if (compiled_without_cluster_path) {
    std::cout << "\nKernel path indicates no cluster code was compiled.\n";
    std::cout << "  Rebuild with a >= SM90 target, e.g.:\n";
    std::cout << "  cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120\n";
#if defined(__CUDA_ARCH_LIST__)
    std::cout
      << "Current __CUDA_ARCH_LIST__ = " << STR(__CUDA_ARCH_LIST__) << "\n";
#endif
  }


  CHECK_CUDA(cudaFree(d_cluster_id));
  CHECK_CUDA(cudaFree(d_block_rank));
  return 0;
#endif
}
