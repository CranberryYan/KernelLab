import triton
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from matmul_opt import matmul_kernel_autotune, get_configs

def benchmark_kernel(config, M, N, K, warmup=10, iters=100):
  x = torch.randn((M, K), device='cuda', dtype=torch.float16)
  w = torch.randn((K, N), device='cuda', dtype=torch.float16)
  out = torch.empty((M, N), device='cuda', dtype=torch.float16)

  grid = lambda META: (
    triton.cdiv(M, META["BLOCK_SIZE_M"]),
    triton.cdiv(N, META["BLOCK_SIZE_N"])
  )

  # Warm-up
  for _ in range(warmup):
    matmul_kernel_autotune[grid](x, w, out, M, N, K)
  torch.cuda.synchronize()

  # Benchmark
  start = time.time()
  for _ in range(iters):
    matmul_kernel_autotune[grid](x, w, out, M, N, K)
  torch.cuda.synchronize()
  end = time.time()

  avg_time_ms = (end - start) * 1000 / iters
  return avg_time_ms

def run_benchmarks(M, N, K):
  configs = get_configs()
  results = []
  for cfg in configs:
    time_ms = benchmark_kernel(cfg, M, N, K)
    print(f"Config {cfg.kwargs} | \
      Warps: {cfg.num_warps}, Stages: {cfg.num_stages} -> {time_ms:.2f} ms")
    results.append({
      "BLOCK_SIZE_M": cfg.kwargs["BLOCK_SIZE_M"],
      "BLOCK_SIZE_N": cfg.kwargs["BLOCK_SIZE_N"],
      "BLOCK_SIZE_K": cfg.kwargs["BLOCK_SIZE_K"],
      "num_warps": cfg.num_warps,
      "num_stages": cfg.num_stages,
      "time_ms": time_ms
    })
  df = pd.DataFrame(results)
  return df

def plot_results(df):
  fig, ax = plt.subplots(figsize=(10, 6))
  labels = [f'\
    M{row["BLOCK_SIZE_M"]}_N{row["BLOCK_SIZE_N"]}_K{row["BLOCK_SIZE_K"]}\
      _w{row["num_warps"]}_s{row["num_stages"]}' 
        for _, row in df.iterrows()]
  ax.bar(labels, df["time_ms"])
  ax.set_ylabel("Avg Execution Time (ms)")
  ax.set_title("Triton Kernel Benchmark by Config")
  ax.set_xticklabels(labels, rotation=45, ha='right')
  plt.tight_layout()
  plt.show()

def main():
  M, N, K = 2048, 2048, 2048
  df = run_benchmarks(M, N, K)
  df.to_csv("triton_benchmark_results.csv", index=False)
  print("Benchmark results saved to triton_benchmark_results.csv")
  plot_results(df)

if __name__ == "__main__":
  main()
