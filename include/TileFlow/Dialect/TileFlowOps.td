#ifndef TILEFLOW_OPS_TD
#define TILEFLOW_OPS_TD

include "TileFlowDialect.td"
include "TileFlowAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// tileflow.matmul
//===----------------------------------------------------------------------===//

def TileFlow_MatmulOp : TileFlow_Op<"matmul", [Pure]> {
  let summary = "Tiled matrix multiplication on ranked tensors";
  let description = [{
    Computes `C = op(A) * op(B)` where `op` is an optional transpose selected
    by `transpose_a` / `transpose_b`. Both inputs are rank-2 tensors of f32 or
    f16 with static shape.

    The iteration space is `(M, N, K)` in that order, so a schedule carries
    three tile sizes and a permutation of `[0, 1, 2]`.

    Example:

    ```mlir
    %0 = tileflow.matmul %A, %B {
        schedule = #tileflow.schedule<tiles = [32, 32, 32], order = [0, 1, 2],
                                      unroll = 1, promote = ["A", "B"],
                                      target = "cpu">
      } : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
    ```
  }];

  let arguments = (ins
    TensorOf<[F32, F16]>:$a,
    TensorOf<[F32, F16]>:$b,
    UnitAttr:$transpose_a,
    UnitAttr:$transpose_b,
    OptionalAttr<TileFlow_ScheduleAttr>:$schedule
  );

  let results = (outs TensorOf<[F32, F16]>:$output);

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    static constexpr int64_t kIterationRank = 3;
    /// Iteration space extents in (M, N, K) order.
    ::llvm::SmallVector<int64_t> getIterationSpaceExtents();
  }];
}

//===----------------------------------------------------------------------===//
// tileflow.softmax_row
//===----------------------------------------------------------------------===//

def TileFlow_SoftmaxRowOp : TileFlow_Op<"softmax_row", [Pure]> {
  let summary = "Row-wise softmax over a rank-2 tensor";
  let description = [{
    Numerically stable softmax along `axis` (default 1) of a rank-2 f32/f16
    tensor. The iteration space is `(rows, columns)`.
  }];

  let arguments = (ins
    TensorOf<[F32, F16]>:$input,
    OptionalAttr<I64Attr>:$axis,
    OptionalAttr<TileFlow_ScheduleAttr>:$schedule
  );

  let results = (outs TensorOf<[F32, F16]>:$output);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static constexpr int64_t kIterationRank = 2;
    ::llvm::SmallVector<int64_t> getIterationSpaceExtents();
    /// Softmax axis, defaulting to 1 when the attribute is absent.
    int64_t getAxisValue() {
      auto attr = getAxisAttr();
      return attr ? attr.getInt() : 1;
    }
  }];
}

//===----------------------------------------------------------------------===//
// tileflow.attention
//===----------------------------------------------------------------------===//

def TileFlow_AttentionOp : TileFlow_Op<"attention", [Pure]> {
  let summary = "Scaled dot-product attention with optional causal masking";
  let description = [{
    Computes `softmax(Q K^T * scale [+ causal mask]) V` for rank-3 tensors
    laid out as `[heads, seq, head_dim]`. Q, K and V must agree on the head
    count; Q and K must agree on the head dimension; K and V must agree on the
    key sequence length. `causal` is only legal when the query and key
    sequence lengths are equal.

    The iteration space is `(heads, seq_q, seq_k, head_dim)`.
  }];

  let arguments = (ins
    TensorOf<[F32, F16]>:$q,
    TensorOf<[F32, F16]>:$k,
    TensorOf<[F32, F16]>:$v,
    DefaultValuedAttr<F64Attr, "1.0">:$scale,
    UnitAttr:$causal,
    OptionalAttr<TileFlow_ScheduleAttr>:$schedule
  );

  let results = (outs TensorOf<[F32, F16]>:$output);

  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    static constexpr int64_t kIterationRank = 4;
    ::llvm::SmallVector<int64_t> getIterationSpaceExtents();
    bool getCausalFlag() { return static_cast<bool>(getCausal()); }
    /// Attention scale, defaulting to 1.0 when the attribute is absent.
    double getScaleValue() {
      auto attr = getScaleAttr();
      return attr ? attr.getValueAsDouble() : 1.0;
    }
  }];
}

//===----------------------------------------------------------------------===//
// tileflow.schedule_region / tileflow.yield
//===----------------------------------------------------------------------===//

def TileFlow_ScheduleRegionOp : TileFlow_Op<"schedule_region", [
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Region whose body is governed by a tiling schedule";
  let description = [{
    Groups a computation under one schedule. Nested regions are collapsed by
    canonicalization by composing their schedules.
  }];

  let arguments = (ins
    TileFlow_ScheduleAttr:$schedule
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

def TileFlow_YieldOp : TileFlow_Op<"yield", [Pure, Terminator,
                                             HasParent<"ScheduleRegionOp">]> {
  let summary = "Terminator for tileflow.schedule_region";
  let arguments = (ins Variadic<AnyRankedTensor>:$values);
  let hasVerifier = 0;
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

#endif // TILEFLOW_OPS_TD
