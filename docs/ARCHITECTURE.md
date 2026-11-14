# Architecture

This document walks through the compiler pass by pass. The IR excerpts are
real `tileflow-opt` output for the causal attention example in
`test/Lowering/` shape (1 head, seq 8, head_dim 8) with schedule
`tiles = [1, 4, 4, 8], order = [0, 1, 2, 3], promote = ["K", "V"]`.

## 1. Dialect layer

`include/TileFlow/Dialect/` contains three TableGen files:

- `TileFlowDialect.td` - namespace, summary, and the `TileFlow_Op` base class.
- `TileFlowAttrs.td` - `#tileflow.schedule<...>` as an `AttrDef` with a custom
  parser/printer and a verifier.
- `TileFlowOps.td` - `MatmulOp`, `SoftmaxRowOp`, `AttentionOp`,
  `ScheduleRegionOp`, `YieldOp`.

`lib/Dialect/TileFlowOps.cpp` implements the custom assembly formats (including
the region parser/printer), op verifiers, and canonicalization. The schedule is
stored as an inherent attribute, so op accessors use `getScheduleAttr()` and
the parser fills the ODS Properties struct directly through
`OperationState::getOrAddProperties`.

Verifier checks are deliberately specific:

- matmul: rank 2, static shape, matching element types, `K` from A equals `K`
  from B, result shape `M x N`, schedule rank 3;
- softmax: rank 2, f32/f16, axis in range, schedule rank 2;
- attention: rank 3, heads agree across Q/K/V, Q/K head dims agree, K/V key
  sequence agree, result shape, causal only when `seq_q == seq_k`, schedule
  rank 4;
- schedule: positive tiles, `order` is a permutation of the iteration rank,
  promotion names are a subset of `{A,B,Q,K,V}`, valid target;
- schedule region: single block, yield count/types match results, schedule rank
  matches the result rank.

Canonicalization is in `lib/Dialect/TileFlowOps.cpp`:

1. `FoldTransposeIntoMatmul` absorbs a 2-D `linalg.transpose` into
   `transpose_a`/`transpose_b`.
2. `InlineIdentityScheduleRegion` splices the body of an identity region into
   its parent.
3. `CollapseNestedScheduleRegions` composes nested schedules (element-wise
   minimum tiles, inner order, multiplied unroll, union of promotions).
4. `CanonicalizeAttentionScale` removes an explicit `scale = 1.0`.
5. The zero-dim matmul pattern replaces an empty result with `tensor.empty`.

## 2. Stage 1: `-tileflow-to-linalg`

`lib/Transforms/TileFlowToLinalg.cpp` lowers each tileflow op with an
`IRRewriter`:

- matmul -> optional `linalg.transpose` + zero-filled `linalg.matmul`;
- softmax -> `linalg.fill` (-sentinel), four `linalg.generic`/`linalg.fill`
  operations implementing max, exp(x - max), sum, and divide;
- attention -> `linalg.transpose` for K^T, zero-filled `linalg.batch_matmul`,
  scale `linalg.generic`, causal-mask `linalg.generic` (using `linalg.index`
  and `arith.select`), softmax chain, then the PV `linalg.batch_matmul`.

Every produced linalg op that participates in the schedule is tagged with the
discardable attribute `tileflow.schedule`. Attention's 4-D schedule is remapped:
QK uses `[H,Sq,Sk,D]`, scale/mask/softmax use `[H,Sq,Sk]`, and PV uses
`[H,Sq,Dv,Sk]` because a batch matmul's iteration space is `[batch,M,N,K]`.

Output (excerpt):

```mlir
%transposed = linalg.transpose ins(%arg1 : tensor<1x8x8xf32>)
    outs(%0 : tensor<1x8x8xf32>) permutation = [0, 2, 1]
%2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
%3 = linalg.batch_matmul {tileflow.schedule = #tileflow.schedule<tiles = [1, 4, 4, 8], order = [0, 1, 2, 3], unroll = 1, promote = ["K", "V"], target = "cpu">}
    ins(%arg0, %transposed : tensor<1x8x8xf32>, tensor<1x8x8xf32>)
    outs(%2 : tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
```

The zero fill matters: `linalg.matmul` accumulates into its destination, so
using `tensor.empty` for the destination produced uninitialized accumulation
during development. The numerical tests caught it.

## 3. Stage 2: `-tileflow-apply-schedule` and the locality passes

`lib/Transforms/ApplySchedule.cpp` implements four passes:

### `-tileflow-tile-for-locality`

Collects linalg ops (scheduled ops by default, all ops when `tile-sizes` is
given) and calls `scf::tileUsingSCF` with `SCFTilingOptions`. Tile sizes and
interchange order come from the schedule or the pass options. The tiling
interface inserts `scf.for` loops, `tensor.extract_slice`/`tensor.insert_slice`,
and `affine.min` boundary sizes for non-divisible extents.

Real output for a 50x70x30 matmul with 16x16x16 tiles:

```mlir
%1 = scf.for %arg2 = %c0 to %c50 step %c16 iter_args(%arg3 = %0) -> (tensor<50x30xf32>) {
  %4 = affine.min #map(%arg2)
  ...
  %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg6] [%4, %6] [1, 1]
      : tensor<50x70xf32> to tensor<?x?xf32>
  %16 = linalg.matmul ins(%extracted_slice, ...) outs(...)
}
```

### `-tileflow-fuse-producers`

Walks the tiled loop nests, finds `tensor.extract_slice` operations whose
source is produced outside the nest, and calls
`scf::tileAndFuseProducerOfSlice`. After fusion it rewrites slices of
`tensor.empty` to tile-sized `tensor.empty` and erases dead producers, so a
full-size elementwise temporary never becomes a `memref.alloc`.

### `-tileflow-promote-to-scratchpad`

After bufferization, each operand that is a static-shaped `memref.subview` is
replaced by a scratchpad buffer: `memref.alloca` for the CPU space, or
`gpu.alloc` with memory space 3 for `space=workgroup`. A `linalg.copy` copies
the slice in before the consumer; destination operands are copied out after
the consumer. Allocations are hoisted to function entry so the stack does not
grow per tile iteration.

### `-tileflow-apply-schedule`

Runs tiling (using schedule tiles/order unless overridden by options), strips
the consumed schedule attributes, optionally fuses producers, and optionally
bufferizes plus promotes the named operands. It is idempotent-safe: the
schedule is gone after the first application, and promotion only matches
subviews that remain.

## 4. Stage 3: `-tileflow-to-loops`

`lib/Transforms/ToLoops.cpp`:

1. If tensor IR remains, runs one-shot bufferization with function-boundary
   bufferization to identity-layout memrefs, using the upstream
   `OneShotBufferizePass`.
2. Runs CSE and removes self-copies (which also keeps `memrefCopy` out of the
   final LLVM module).
3. When `vectorize=true`, vectorizes elementwise `linalg.generic` ops with
   static shapes up to 1024 elements. Reductions and matmuls are skipped
   because their multi-dimensional vector forms were not lowerable at these
   tile sizes.
4. Runs `-convert-linalg-to-loops`.
5. Optionally unrolls innermost constant-trip-count loops.

Stage-3 output is scalar memref code:

```mlir
%0 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %alloc) -> (memref<1x8x8xf32>) {
  %8 = scf.for %arg5 = %c0 to %c8 step %c4 iter_args(%arg6 = %arg4) -> (memref<1x8x8xf32>) {
    ...
    %11 = memref.load %subview[%arg11, %arg12, %arg14]
    %12 = memref.load %alloc_10[%arg11, %arg14, %arg13]
    %14 = arith.mulf %11, %12 : f32
    %15 = arith.addf %13, %14 : f32
    memref.store %15, %subview_11[%arg11, %arg12, %arg13]
```

## 5. Stage 4: `-tileflow-to-llvm`

`lib/Transforms/ToLLVM.cpp` runs, in order:

```
lower-affine
convert-vector-to-scf
convert-scf-to-cf
convert-vector-to-llvm
expand-strided-metadata
lower-affine                      (expand-strided-metadata emits affine.apply)
finalize-memref-to-llvm
lower-affine
arith-to-llvm
math-to-llvm
index-to-llvm
control-flow-to-llvm
ub-to-llvm
func-to-llvm
reconcile-unrealized-casts
```

The result contains only the LLVM dialect (plus `builtin.module`), and
`mlir-translate --mlir-to-llvmir` accepts it. `test/Lowering/stage4-to-llvm.mlir`
checks both properties.

## 6. GPU branch: `-tileflow-lower-to-gpu`

`lib/Transforms/Pipelines.cpp`:

1. Lowers tileflow to linalg.
2. Tiles with `SCFTilingOptions::LoopType::ForallOp` and GPU mappings. Reduction
   loops get tile size 0 so they are not distributed across threads.
3. Bufferizes, lowers linalg to loops, converts `scf.forall` to `scf.parallel`,
   and lowers each parallel loop to `gpu.launch`. One parallel iteration maps
   to one GPU block; induction variables are reconstructed from block IDs.
4. For `stop-after=nvvm`, outlines the launch into a `gpu.module`, converts the
   kernel body to NVVM (`convert-gpu-ops-to-nvvm`) and leaves a
   `gpu.launch_func` host call.

```mlir
%1 = scf.forall (%arg2, %arg3) = (0, 0) to (64, 64) step (16, 16)
    shared_outs(%arg4 = %0) -> (tensor<64x64xf32>) {
  ...
} {mapping = [#gpu.block<x>, #gpu.thread<x>]}
```

At `stop-after=nvvm` the module contains a `gpu.module @matmul_kernel` with
`nvvm.read.ptx.sreg.ctaid.x` and a `gpu.launch_func`. No device is present on
the build machine, so this path is compile/FileCheck tested only.

## 7. Cost model

`include/TileFlow/Analysis/LocalityCostModel.h` and
`lib/Analysis/LocalityCostModel.cpp`:

- input: extents, per-operand loop dependencies and element size, FLOP count,
  schedule, machine description;
- block working set: sum of operand tile bytes at the innermost level;
- per cache level, a reuse loop's multiplier is its tile trip count when the
  block fits the cache, otherwise the loop's full extent (data is re-streamed);
- traffic per level is `distinctTiles * tileBytes * product(load factors)`,
  clipped to the total element references;
- DRAM traffic is the traffic that misses the last cache;
- reuse distance for each `(operand, loop it does not index)` is the bytes
  touched by the loops inside that loop; distances are averaged and bucketed by
  power of two;
- promoted scratchpad footprint is checked against `scratchpad_bytes`;
  over-capacity schedules are infeasible.

`-tileflow-locality-report` serializes all of this as JSON; `tileflow-tune`
ranks candidates with it and annotates the winner back into the IR.
