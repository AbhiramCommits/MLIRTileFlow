# MLIR-TileFlow

**MLIR-TileFlow** is a custom MLIR dialect for tiled tensor operations (matmul,
row softmax, and fused attention) plus a staged lowering pipeline to LLVM. It
ships real data-locality passes (loop tiling, producer fusion, scratchpad
promotion) and an analytical cost model that chooses tile sizes from cache
capacity and reuse distance instead of compiling and timing every candidate.

The pipeline is real MLIR: TableGen-generated ops and passes, C++17, out-of-tree
CMake against an installed LLVM/MLIR, lit/FileCheck tests, and kernels that run
through `ctypes` after `mlir-translate` + `clang`.

---

## Architecture

```
                        tileflow dialect (tensor, schedule attr)
                                        |
                     -tileflow-to-linalg |  linalg on tensors
                                        v
                  linalg.matmul / batch_matmul / generic + tileflow.schedule
                                        |
            -tileflow-apply-schedule     |  tiling, producer fusion, bufferize,
            (or the three standalone     |  scratchpad promotion
             locality passes)            v
                     scf.for (or scf.forall for the GPU branch) + memref
                                        |
                     -tileflow-to-loops  |  one-shot bufferize, linalg -> scf,
                                        v  optional vectorization, unrolling
                     affine / scf / vector loops on memrefs
                                        |
                     -tileflow-to-llvm   |  affine->scf->cf, vector/memref/
                                        v  arith/math/index/func -> LLVM
                                  LLVM dialect

GPU branch:  schedule target="gpu"
   linalg -> scf.forall [#gpu.block/#gpu.thread] -> gpu.launch
           -> gpu.module kernel -> NVVM  (-tileflow-lower-to-gpu)
```

Each stage is an independently invocable pass and has its own FileCheck test.
`-tileflow-lower='stop-after=...'` runs the composite pipeline up to
`linalg`, `schedule`, `loops`, or `llvm`, which makes the intermediate IR of
any stage available for inspection without editing the input.

### Operations

| Operation | Syntax | Semantics |
|---|---|---|
| `tileflow.matmul` | `%c = tileflow.matmul %a, %b {transpose_a, transpose_b, schedule}` | rank-2 f32/f16 matmul; iteration space `(M, N, K)` |
| `tileflow.softmax_row` | `%y = tileflow.softmax_row %x {axis = 1, schedule}` | stable row softmax on a rank-2 tensor; iteration space `(rows, cols)` |
| `tileflow.attention` | `%o = tileflow.attention %q, %k, %v {scale, causal, schedule}` | `softmax(QK^T * scale [+ mask]) V` on `[heads, seq, head_dim]`; iteration space `(heads, seq_q, seq_k, head_dim)` |
| `tileflow.schedule_region` | `tileflow.schedule_region #tileflow.schedule<...> { ... }` | groups a body under one schedule; nested regions are collapsed |
| `tileflow.yield` | `tileflow.yield %v : type` | terminator for `schedule_region` |

The schedule attribute is first-class and round-trips through the parser:

```mlir
#tileflow.schedule<tiles = [32, 32, 64], order = [0, 1, 2], unroll = 2,
                    promote = ["A", "B"], target = "cpu">
```

Verifiers reject K-dimension mismatches, inconsistent attention head/sequence
shapes, causal masking with `seq_q != seq_k`, schedule rank mismatches,
non-positive tiles, non-permutation loop orders, unknown promotion names, and
bad targets. Canonicalization folds `linalg.transpose` into `transpose_a/b`,
removes identity schedule regions, collapses nested regions, drops redundant
`scale = 1.0`, and folds zero-dimension matmuls.

### Passes

Deeper pass-by-pass notes with IR at every stage are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); generated pass
documentation is in [`docs/Passes.md`](docs/Passes.md)
(produced by `mlir-tblgen -gen-pass-doc`).

| Pass | Options | What it does |
|---|---|---|
| `-tileflow-to-linalg` | | lowers tileflow ops to linalg and tags produced ops with `tileflow.schedule` |
| `-tileflow-apply-schedule` | `tile-sizes`, `order`, `fuse`, `promote`, `space` | consumes schedules; tiles, fuses producers, bufferizes, promotes tiles |
| `-tileflow-to-loops` | `vectorize`, `unroll` | bufferizes if needed, vectorizes elementwise generics, lowers linalg to loops |
| `-tileflow-to-llvm` | | lowers everything down to the LLVM dialect |
| `-tileflow-tile-for-locality` | `tile-sizes`, `order` | standalone tiling of scheduled linalg ops, including boundary tiles |
| `-tileflow-fuse-producers` | | fuses producer slices into tiled loop nests and removes full-size temporaries |
| `-tileflow-promote-to-scratchpad` | `space`, `operands` | copies operand tiles into `memref.alloca` or `gpu.alloc` (space 3), with copy-in/copy-out |
| `-tileflow-locality-report` | `machine` | evaluates schedules and prints the cost model as JSON |
| `-tileflow-lower` | `stop-after` | composite CPU pipeline: `linalg`, `schedule`, `loops`, `llvm` |
| `-tileflow-lower-to-gpu` | `tile-sizes`, `stop-after` | `forall` -> `gpu` -> `nvvm` GPU branch |

### Tree

```
include/TileFlow/Dialect/     TileFlowDialect.td, TileFlowAttrs.td, TileFlowOps.td + headers
lib/Dialect/                  schedule attribute, op verifiers, canonicalization
include/TileFlow/Transforms/  Passes.td, Passes.h, Utils.h
lib/Transforms/               lowering, tiling/fusion/promotion, LLVM and GPU pipelines
include/TileFlow/Analysis/    LocalityCostModel.h
lib/Analysis/                 analytical model + JSON report pass
tools/tileflow-opt/           mlir-opt-compatible driver
tools/tileflow-tune/          cost-model schedule selector / annotator
test/                         lit tests: Dialect, Lowering, Transforms, Analysis
python/                       ctypes kernel builder + PyTorch numerical validation
bench/                        machine description, benchmark/tuning inputs and runners
docs/                         toolchain record, architecture and generated pass docs
results/                      measured and model-estimated outputs committed as evidence
```

### Cost model

`lib/Analysis/LocalityCostModel.cpp` takes:

- iteration-space extents and, per operand, which loops index it (access maps),
- FLOP count,
- a candidate schedule (`tiles`, `order`, `unroll`, `promote`, `target`),
- a machine description: cache/SRAM capacities, line size, vector width
  (`bench/machine.json`).

It produces, analytically and without compiling or timing:

- bytes moved at each cache level and the estimated DRAM traffic, using
  block-working-set-versus-capacity reasoning per loop level,
- arithmetic intensity (FLOPs / DRAM byte),
- average and histogrammed reuse distance per operand,
- scratchpad footprint for promoted operands, and whether the schedule is
  feasible (a promoted tile larger than the scratchpad is rejected).

`tileflow-tune` enumerates a bounded power-of-two tile space, all loop orders,
and promotion on/off, ranks the candidates **only by the model**, writes
`results/cost_model_ranking.json`, and can annotate the winning schedule back
into MLIR with `--annotate in.mlir -o out.mlir`.

---

## How to run

### Toolchain and build

The build used LLVM/MLIR **23.1.1** from Homebrew on macOS. The record produced
by the bootstrap script is in [`docs/toolchain.txt`](docs/toolchain.txt).

```bash
# from a clean checkout
python3 -m venv .venv
.venv/bin/pip install lit pytest numpy torch
./scripts/bootstrap.sh          # resolves MLIR_DIR and prints tool paths
MLIR_DIR=$(./scripts/bootstrap.sh | sed -n 's/^MLIR_DIR=//p')
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR="$MLIR_DIR"
cmake --build build -j
```

The project compiles with `-Wall -Wextra` and produced zero warnings in the
recorded clean build (`results/build_log.txt`).

`scripts/bootstrap.sh` tries `$MLIR_DIR`, `$(brew --prefix llvm)/lib/cmake/mlir`,
`/usr/lib/llvm-*/lib/cmake/mlir`, and `llvm-config --cmakedir`, and exits with
apt.llvm.org/Homebrew instructions if MLIR is missing. It prints the resolved
MLIR_DIR, LLVM version, and `mlir-tblgen`, `mlir-opt`, `mlir-translate`, `lit`
paths.

### Stage by stage

```bash
# 1. tileflow -> linalg (schedule is carried as a discardable attribute)
./build/bin/tileflow-opt -tileflow-to-linalg input.mlir

# 2. apply the schedule: tile, fuse, bufferize, promote
./build/bin/tileflow-opt -tileflow-to-linalg \
    -tileflow-apply-schedule='fuse=true promote=true' input.mlir

# 3. loops and vectorization
./build/bin/tileflow-opt -tileflow-to-linalg -tileflow-apply-schedule \
    -tileflow-to-loops='vectorize=true' input.mlir

# 4. LLVM dialect
./build/bin/tileflow-opt -tileflow-to-linalg -tileflow-apply-schedule \
    -tileflow-to-loops -tileflow-to-llvm input.mlir

# composite with early stop
./build/bin/tileflow-opt -tileflow-lower='stop-after=loops' input.mlir
```

Real stage-1 output for a causal 1x8x8 attention (schedule
`tiles=[1,4,4,8]`, `promote=["K","V"]`):

```mlir
%transposed = linalg.transpose ins(%arg1 : tensor<1x8x8xf32>)
    outs(%0 : tensor<1x8x8xf32>) permutation = [0, 2, 1]
%2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
%3 = linalg.batch_matmul {tileflow.schedule = #tileflow.schedule<tiles = [1, 4, 4, 8], order = [0, 1, 2, 3], unroll = 1, promote = ["K", "V"], target = "cpu">}
    ins(%arg0, %transposed : tensor<1x8x8xf32>, tensor<1x8x8xf32>)
    outs(%2 : tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
```

Stage 2 produces the tiled tensor loop (excerpt):

```mlir
%22 = scf.for %arg9 = %c0_2 to %c8_4 step %c8_7 iter_args(%arg10 = %arg8) -> (tensor<1x8x8xf32>) {
  %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5, %arg9] [1, 4, 8] [1, 1, 1] : tensor<1x8x8xf32> to tensor<1x4x8xf32>
  %transposed = linalg.transpose ins(%extracted_slice_79 : tensor<1x4x8xf32>) outs(%23 : tensor<1x8x4xf32>) permutation = [0, 2, 1]
  %24 = linalg.batch_matmul ins(%extracted_slice, %transposed : tensor<1x4x8xf32>, tensor<1x8x4xf32>) outs(%extracted_slice_80 : tensor<1x4x4xf32>)
  %inserted_slice = tensor.insert_slice %24 into %arg10[%arg3, %arg5, %arg7] [1, 4, 4] [1, 1, 1]
  scf.yield %inserted_slice : tensor<1x8x8xf32>
}
```

Stage 3 bufferizes and lowers that to scalar loops:

```mlir
%0 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %alloc) -> (memref<1x8x8xf32>) {
  %8 = scf.for %arg5 = %c0 to %c8 step %c4 iter_args(%arg6 = %arg4) -> (memref<1x8x8xf32>) {
    %9 = scf.for %arg7 = %c0 to %c8 step %c4 iter_args(%arg8 = %arg6) -> (memref<1x8x8xf32>) {
      ...
      %11 = memref.load %subview[%arg11, %arg12, %arg14] : memref<1x4x8xf32, strided<[64, 8, 1], offset: ?>>
      %12 = memref.load %alloc_10[%arg11, %arg14, %arg13] : memref<1x8x4xf32>
      %14 = arith.mulf %11, %12 : f32
      %15 = arith.addf %13, %14 : f32
      memref.store %15, %subview_11[%arg11, %arg12, %arg13] : memref<1x4x8xf32, strided<[64, 8, 1], offset: ?>>
```

Stage 4 is a pure LLVM-dialect module:

```mlir
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @attention(%arg0: !llvm.ptr, ..., %arg26: i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
```

### Tests

```bash
cmake --build build --target check-tileflow     # lit + FileCheck
./.venv/bin/python -m pytest python -q          # PyTorch numerical validation
ctest --test-dir build --output-on-failure      # both of the above
```

### Tuning and benchmarking

```bash
./build/bin/tileflow-tune bench/matmul_1024.mlir --machine bench/machine.json
./build/bin/tileflow-tune bench/matmul_1024.mlir --machine bench/machine.json \
    --annotate bench/matmul_1024.mlir -o /tmp/annotated.mlir
./build/bin/tileflow-opt -tileflow-to-linalg -tileflow-apply-schedule \
    /tmp/annotated.mlir

.venv/bin/python bench/run_bench.py
.venv/bin/python bench/run_bench_gpu.py
```

`run_bench.py` compiles the untiled, tiled+fused, and
tiled+fused+promoted+vectorized variants for each shape, warms up, times 20
repetitions with `time.perf_counter` (median, p10, p90), computes GFLOP/s, and
pairs each variant with the cost model's predicted DRAM traffic. It also
compiles a candidate set and writes the Spearman correlation and top-1 regret.

---

## Results

All figures below were produced by commands in this repository on the machine
recorded in `docs/toolchain.txt`; the raw outputs are committed under
`results/`.

**Toolchain and machine.** LLVM/MLIR **23.1.1** (Homebrew), macOS 27.2.1 on
Apple silicon, **Apple M3**, 8 cores (4 performance + 4 efficiency), 16 GB
memory (`results/machine.txt`). Cache description used by the model
(`bench/machine.json`, from `sysctl`): L1D 65,536 B, L2 4,194,304 B, 128-byte lines, 16 GB DRAM,
65,536 B scratchpad budget.

**Tests.** 16 lit/FileCheck tests pass (`results/lit_output.txt`), 7 pytest
numerical tests pass (`results/pytest_output.txt`), and
`ctest --output-on-failure` reports 2/2 tests passed (`results/test_output.txt`).

### Numerical validation against PyTorch eager

Compiled kernels are compared with `torch.matmul` and
`torch.nn.functional.scaled_dot_product_attention` (f32; atol=2e-3,
rtol=2e-3). All three pipeline variants produced identical maxima for these
schedules; the table reports the measured maximum over the three from
`results/numerics.json`.

| case | max abs error | max relative error |
|---|---:|---:|
| matmul 256x256x256 | 1.431e-06 | 6.918e-03 |
| matmul 1024x1024x1024 | 3.338e-06 | 5.538e-02 |
| matmul 513x257x129 (non-divisible) | 7.153e-07 | 4.362e-03 |
| attention seq=128, head_dim=64 | 1.192e-07 | 8.198e-04 |
| attention seq=128 causal | 1.788e-07 | 2.085e-03 |
| attention seq=512, head_dim=64 | 9.406e-08 | 8.314e-03 |
| attention seq=512 causal | 1.192e-07 | 2.270e-03 |

The largest relative values come from elements near zero (the denominator is
clipped at 1e-6); absolute errors stay below 3.4e-6, and every case passes
`numpy.allclose` with atol=2e-3, rtol=2e-3.

### CPU benchmarks (measured) and model traffic (estimated)

Measured with `time.perf_counter`, 20 repetitions, median; the machine-readable
source is `results/bench_cpu.csv` and the Markdown copy is
`results/bench_cpu.md`. `predicted DRAM` is **model-estimated**, not measured.

| shape | variant | median ms | p10 ms | p90 ms | GFLOP/s | speedup vs untiled | predicted DRAM bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| matmul_256 | untiled | 14.170 | 13.790 | 14.819 | 2.37 | 1.00x | 786,432 |
| matmul_256 | tiled+fused | 8.107 | 7.791 | 8.395 | 4.14 | 1.75x | 3,145,728 |
| matmul_256 | tiled+fused+promoted+vectorized | 7.713 | 7.476 | 8.191 | 4.35 | **1.84x** | 3,145,728 |
| matmul_1024 | untiled | 1335.999 | 1275.773 | 1459.160 | 1.61 | 1.00x | 12,884,901,888 |
| matmul_1024 | tiled+fused | 1138.192 | 1105.312 | 1187.221 | 1.89 | 1.17x | 201,326,592 |
| matmul_1024 | tiled+fused+promoted+vectorized | 450.919 | 444.377 | 486.725 | 4.76 | **2.96x** | 201,326,592 |
| matmul_nondivisible | untiled | 9.473 | 9.254 | 9.878 | 3.59 | 1.00x | 924,684 |
| matmul_nondivisible | tiled+fused | 9.450 | 9.228 | 9.599 | 3.60 | 1.00x | 4,675,968 |
| matmul_nondivisible | tiled+fused+promoted+vectorized | 9.433 | 9.200 | 9.806 | 3.61 | 1.00x | 4,675,968 |
| attention_seq128 | untiled | 1.094 | 1.089 | 1.183 | 3.91 | 1.00x | 196,608 |
| attention_seq128 | tiled+fused | 0.610 | 0.600 | 0.658 | 7.02 | 1.79x | 393,216 |
| attention_seq128 | tiled+fused+promoted+vectorized | 0.607 | 0.602 | 0.633 | 7.04 | 1.80x | 393,216 |
| attention_seq512 | untiled | 21.311 | 21.179 | 21.676 | 3.21 | 1.00x | 1,572,864 |
| attention_seq512 | tiled+fused | 10.158 | 9.935 | 10.376 | 6.74 | **2.10x** | 6,291,456 |
| attention_seq512 | tiled+fused+promoted+vectorized | 10.515 | 10.214 | 11.121 | 6.51 | 2.03x | 6,291,456 |

Headline: the promoted variant is **2.96x** faster than untiled at
1024x1024x1024 (450.919 ms vs 1335.999 ms) and **1.84x** at 256x256x256; plain
tiling is 1.17x at 1024 and 1.75x at 256. Non-divisible tiling is neutral
(1.00x for all variants): boundary-tile overhead cancels the locality gain at
this size. Attention gains 1.79x and 2.10x at seq=128/512.

### Cost model

For the 1024 matmul the tuner selected
`tiles = [64, 64, 64]`, `order = [0, 1, 2]`, no promotion. Its
model-estimated DRAM traffic is **201,326,592 bytes** versus
**12,884,901,888 bytes** for the untiled full-tile schedule, a **64.00x**
reduction (model estimate). Predicted arithmetic intensity is **10.67
FLOP/byte** and the average reuse distance is 4,472,832 bytes
(`results/tune_log.txt`, `results/cost_model_ranking.json`).

Tuning cost: **2,592** candidates enumerated and ranked by the model,
**2,064** passed the scratchpad feasibility check (**528** pruned without
compiling), versus **2,592** that brute-force autotuning would have compiled.
These counts are in `results/tune_log.txt`.

### Model versus measured

On a 512x512x512 matmul with 8 curated candidate schedules (untiled through
32x32x32, including mixed tilings):

- Spearman correlation between predicted traffic and measured median runtime:
  **-0.878**
- top-1 regret `(runtime(model top-1) / runtime(best measured)) - 1`:
  **10.286**

Both are in `results/model_vs_measured.json`. The model is anti-correlated with
measured runtime on this candidate set: its top pick is the untiled schedule
(3,145,728 predicted bytes, 117.8 ms) while 32x32x32 tiles moved more predicted
bytes (50,331,648) but ran in 10.4 ms because the 12 KB block fits L1 and the
generated inner loops vectorize well. The model measures bytes, not cache
residency benefits or generated-code quality; this is a real limitation, not a
tuning artifact. `perf stat` with cache/DRAM counters is not available on
macOS, so **all traffic numbers are model-estimated and only runtime is
measured** — the JSON records `perf_stat_available: false` and
`llc_misses_measured: false`.

### GPU status

**No CUDA device is present.** `torch.cuda.is_available()` is false;
`bench/run_bench_gpu.py` therefore wrote `results/gpu_status.txt` and exited
cleanly, and no GPU execution numbers are reported. The GPU lowering itself is
validated by `test/Lowering/gpu-lowering.mlir`, which FileChecks
`scf.forall` with `#gpu.block<...>/#gpu.thread<...>` mappings, `gpu.launch`,
a `gpu.module` kernel, and NVVM ops (`nvvm.read.ptx.sreg.ctaid.x`,
`gpu.launch_func`). It was compile-tested only.

---

## Limitations and next steps

- **Shapes/dtypes.** tileflow ops require ranked tensors with static shapes and
  f32/f16 elements. Dynamic shapes, broadcasting, batches beyond the leading
  head dimension of attention, and non-f32/f16 types are not supported.
- **Attention scope.** Only dense non-flash attention with an optional causal
  mask and a scalar scale; no dropout, ALiBi, sliding windows, GQA/MQA, or
  paged KV caches.
- **Cost model.** It estimates DRAM/level traffic from working-set sizes and
  reuse distances. It ignores cache associativity, line-level conflict misses,
  loop overhead, instruction count, vectorization efficiency, bandwidth
  saturation, and multi-core interference. It ranks total predicted bytes
  moved, so it can mis-rank schedules when small tiles move more bytes but use
  L1 and vectorized inner loops better (the negative Spearman result above is
  an example). It does not model GPU
  occupancy or shared-memory bank conflicts.
- **Vectorization.** The vectorized variant vectorizes elementwise linalg
  generics only; full matmul vectorization produced multi-dimensional
  reductions that the LLVM lowering could not handle at these tile sizes, so
  matmul tiles are not vectorized in the measured "vectorized" variant.
- **GPU.** The GPU path is lowering-verified only (no device on this machine).
  There is no CUDA launch harness in `bench/run_bench_gpu.py`, and no GPU
  performance claim is made.
- **Fusion.** Producer fusion removes full-size elementwise temporaries and
  tile-level fused producers, but the attention softmax reductions are not
  fully fused into the QK^T tile loop; the reduction temporaries still exist at
  tile size.
- **Next steps.** Add dynamic-shape support, a proper vector-tile contract for
  matmul, cost-model terms for L1 latency and instruction overhead (to fix the
  small-shape ranking), a CUDA launch runner, and flash-attention-style online
  softmax.

---

## License

MIT. See [`LICENSE`](LICENSE).
