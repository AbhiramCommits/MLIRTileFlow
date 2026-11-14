# CPU benchmarks

Machine: macOS-27.2.1-arm64-arm-64bit-Mach-O / arm

| shape | variant | median ms | p10 ms | p90 ms | GFLOP/s | speedup | predicted DRAM bytes (model) |
|---|---|---:|---:|---:|---:|---:|---:|
| matmul_256 | untiled | 14.170 | 13.790 | 14.819 | 2.37 | 1.00x | 786432 |
| matmul_256 | tiled_fused | 8.107 | 7.791 | 8.395 | 4.14 | 1.75x | 3145728 |
| matmul_256 | tiled_fused_promoted_vectorized | 7.713 | 7.476 | 8.191 | 4.35 | 1.84x | 3145728 |
| matmul_1024 | untiled | 1335.999 | 1275.773 | 1459.160 | 1.61 | 1.00x | 12884901888 |
| matmul_1024 | tiled_fused | 1138.192 | 1105.312 | 1187.221 | 1.89 | 1.17x | 201326592 |
| matmul_1024 | tiled_fused_promoted_vectorized | 450.919 | 444.377 | 486.725 | 4.76 | 2.96x | 201326592 |
| matmul_nondivisible | untiled | 9.473 | 9.254 | 9.878 | 3.59 | 1.00x | 924684 |
| matmul_nondivisible | tiled_fused | 9.450 | 9.228 | 9.599 | 3.60 | 1.00x | 4675968 |
| matmul_nondivisible | tiled_fused_promoted_vectorized | 9.433 | 9.200 | 9.806 | 3.61 | 1.00x | 4675968 |
| attention_seq128 | untiled | 1.094 | 1.089 | 1.183 | 3.91 | 1.00x | 196608 |
| attention_seq128 | tiled_fused | 0.610 | 0.600 | 0.658 | 7.02 | 1.79x | 393216 |
| attention_seq128 | tiled_fused_promoted_vectorized | 0.607 | 0.602 | 0.633 | 7.04 | 1.80x | 393216 |
| attention_seq512 | untiled | 21.311 | 21.179 | 21.676 | 3.21 | 1.00x | 1572864 |
| attention_seq512 | tiled_fused | 10.158 | 9.935 | 10.376 | 6.74 | 2.10x | 6291456 |
| attention_seq512 | tiled_fused_promoted_vectorized | 10.515 | 10.214 | 11.121 | 6.51 | 2.03x | 6291456 |

`predicted DRAM bytes` is produced by the analytical cost model;
runtime columns are measured with `time.perf_counter` (median of 20).
Vectorization applies to elementwise linalg generics only; matmul
vectorization was intentionally skipped for these tile sizes.
