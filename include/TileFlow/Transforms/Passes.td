#ifndef TILEFLOW_TRANSFORMS_PASSES_TD
#define TILEFLOW_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Stage 1: tileflow -> linalg
//===----------------------------------------------------------------------===//

def TileFlowToLinalg : Pass<"tileflow-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Lower tileflow ops to linalg on tensors";
  let description = [{
    Lowers `tileflow.matmul`, `tileflow.softmax_row` and
    `tileflow.attention` to equivalent linalg ops. The schedule attached to
    the source op is copied to every produced linalg op as the discardable
    attribute `tileflow.schedule`, where the scheduling stages pick it up.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::math::MathDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tileflow::TileFlowDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Stage 2: apply a schedule
//===----------------------------------------------------------------------===//

def TileFlowApplySchedule : Pass<"tileflow-apply-schedule", "::mlir::ModuleOp"> {
  let summary = "Tile, fuse and promote according to tileflow.schedule attributes";
  let description = [{
    Consumes `tileflow.schedule` attributes on linalg ops. Unless overridden
    from the command line, the tile sizes, loop order and promotion set come
    from the attribute. When `fuse` is set, producer fusion runs before
    bufferization; when `promote` is set, the selected operand tiles are
    copied into a function-local scratchpad after bufferization.
  }];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Override the schedule tile sizes">,
    ListOption<"loopOrder", "order", "int64_t",
               "Override the schedule loop order">,
    Option<"fuse", "fuse", "bool", /*default=*/"true",
           "Fuse producers into the tiled consumer loop nest">,
    Option<"promote", "promote", "bool", /*default=*/"false",
           "Promote operand tiles into a scratchpad">,
    Option<"space", "space", "std::string", /*default=*/"\"private\"",
           "Memory space used for scratchpad promotion">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Stage 3: loops and vectorization
//===----------------------------------------------------------------------===//

def TileFlowToLoops : Pass<"tileflow-to-loops", "::mlir::ModuleOp"> {
  let summary = "Bufferize and lower scheduled linalg ops to loops";
  let description = [{
    Runs one-shot bufferization (if tensor IR is still present), optionally
    vectorizes linalg ops, then lowers linalg to `scf.for` loops. The
    `unroll` option unrolls innermost loops with a constant trip count.
  }];
  let options = [
    Option<"vectorize", "vectorize", "bool", /*default=*/"false",
           "Vectorize linalg ops before lowering to loops">,
    Option<"unroll", "unroll", "int64_t", /*default=*/"1",
           "Unroll innermost constant-trip-count loops by this factor">
  ];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Stage 4: LLVM dialect
//===----------------------------------------------------------------------===//

def TileFlowToLLVM : Pass<"tileflow-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lower scheduled loops all the way to the LLVM dialect";
  let description = [{
    Runs affine -> scf -> cf, vector -> llvm, memref -> llvm, arith/math/
    index -> llvm, func -> llvm and reconciles unrealized casts. The result
    contains only the LLVM dialect (plus the builtin module).
  }];
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::func::FuncDialect",
    "::mlir::ub::UBDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Data-locality transformation passes
//===----------------------------------------------------------------------===//

def TileFlowTileForLocality : Pass<"tileflow-tile-for-locality",
                                    "::mlir::ModuleOp"> {
  let summary = "Tile scheduled linalg ops for data locality";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes (overrides the schedule attribute)">,
    ListOption<"loopOrder", "order", "int64_t",
               "Loop interchange order (overrides the schedule attribute)">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def TileFlowFuseProducers : Pass<"tileflow-fuse-producers",
                                  "::mlir::ModuleOp"> {
  let summary = "Fuse producers into tiled consumer loop nests";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def TileFlowPromoteToScratchpad : Pass<"tileflow-promote-to-scratchpad",
                                       "::mlir::ModuleOp"> {
  let summary = "Promote tiled operand slices into a scratchpad buffer";
  let options = [
    Option<"space", "space", "std::string", /*default=*/"\"private\"",
           "Memory space for the scratchpad (private or workgroup)">,
    ListOption<"operands", "operands", "std::string",
               "Operand names to promote (A, B, Q, K, V)">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Cost model report
//===----------------------------------------------------------------------===//

def TileFlowLocalityReport : Pass<"tileflow-locality-report",
                                   "::mlir::ModuleOp"> {
  let summary = "Print the analytical data-locality cost report as JSON";
  let description = [{
    Evaluates every scheduled tileflow op with the analytical locality cost
    model and prints one JSON object per op: bytes moved per memory level,
    estimated DRAM traffic, arithmetic intensity, reuse-distance statistics
    and scratchpad feasibility.
  }];
  let options = [
    Option<"machinePath", "machine", "std::string", /*default=*/"\"\"",
           "Path to a machine description JSON file">
  ];
  let dependentDialects = ["::mlir::tileflow::TileFlowDialect"];
}

//===----------------------------------------------------------------------===//
// Composite pipelines
//===----------------------------------------------------------------------===//

def TileFlowLower : Pass<"tileflow-lower", "::mlir::ModuleOp"> {
  let summary = "Composite tileflow lowering pipeline";
  let options = [
    Option<"stopAfter", "stop-after", "std::string", /*default=*/"\"llvm\"",
           "Stop after one of: linalg, schedule, loops, llvm">
  ];
}

def TileFlowLowerToGPU : Pass<"tileflow-lower-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower tileflow through scf.forall and the GPU/NVVM path";
  let options = [
    Option<"tileSizes", "tile-sizes", "std::string",
           /*default=*/"\"\"", "Comma-separated tile-size override">,
    Option<"stopAfter", "stop-after", "std::string", /*default=*/"\"gpu\"",
           "Stop after one of: forall, gpu, nvvm">
  ];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::NVVM::NVVMDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect"
  ];
}

#endif // TILEFLOW_TRANSFORMS_PASSES_TD
