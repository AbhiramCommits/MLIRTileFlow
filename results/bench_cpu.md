# CPU benchmarks

Machine: macOS-27.2.1-arm64-arm-64bit-Mach-O / arm

| shape | variant | median ms | p10 ms | p90 ms | GFLOP/s | speedup | predicted DRAM bytes (model) |
|---|---|---:|---:|---:|---:|---:|---:|
| matmul_256 | untiled | 23.100 | 20.155 | 25.364 | 1.45 | 1.00x | 786432 |
| matmul_256 | tiled_fused | 22.340 | 20.132 | 26.174 | 1.50 | 1.03x | 786432 |
| matmul_256 | tiled_fused_promoted_vectorized | 14.688 | 14.256 | 16.861 | 2.28 | 1.57x | 2621440 |
| matmul_1024 | untiled | 2269.195 | 2088.985 | 2630.559 | 0.95 | 1.00x | 12884901888 |
| matmul_1024 | tiled_fused | 2025.236 | 1649.657 | 2254.319 | 1.06 | 1.12x | 71303168 |
| matmul_1024 | tiled_fused_promoted_vectorized | 632.391 | 588.406 | 727.408 | 3.40 | 3.59x | 167772160 |
| matmul_nondivisible | untiled | 10.903 | 10.620 | 11.505 | 3.12 | 1.00x | 924684 |
| matmul_nondivisible | tiled_fused | 12.144 | 11.550 | 13.070 | 2.80 | 0.90x | 924684 |
| matmul_nondivisible | tiled_fused_promoted_vectorized | 9.811 | 9.315 | 11.858 | 3.47 | 1.11x | 5898240 |
| attention_seq128 | untiled | 1.208 | 1.198 | 1.311 | 3.54 | 1.00x | 196608 |
| attention_seq128 | tiled_fused | 1.207 | 1.201 | 1.251 | 3.54 | 1.00x | 196608 |
| attention_seq128 | tiled_fused_promoted_vectorized | 1.324 | 1.284 | 1.833 | 3.23 | 0.91x | 196608 |
| attention_seq512 | untiled | 28.286 | 27.605 | 31.582 | 2.42 | 1.00x | 1572864 |
| attention_seq512 | tiled_fused | 27.856 | 26.015 | 36.240 | 2.46 | 1.02x | 1572864 |
| attention_seq512 | tiled_fused_promoted_vectorized | 23.936 | 22.906 | 33.604 | 2.86 | 1.18x | 3145728 |

`predicted DRAM bytes` is produced by the analytical cost model;
runtime columns are measured with `time.perf_counter` (median of 20).
Vectorization applies to elementwise linalg generics only; matmul
vectorization was intentionally skipped for these tile sizes.
