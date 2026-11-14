#include "TileFlow/Transforms/Passes.h"
#include "TileFlow/Transforms/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace tileflow {
#define GEN_PASS_DEF_TILEFLOWAPPLYSCHEDULE
#define GEN_PASS_DEF_TILEFLOWTILEFORLOCALITY
#define GEN_PASS_DEF_TILEFLOWFUSEPRODUCERS
#define GEN_PASS_DEF_TILEFLOWPROMOTETOSCRATCHPAD
#include "TileFlow/Transforms/Passes.h.inc"
} // namespace tileflow
} // namespace mlir

using namespace mlir;
using namespace mlir::tileflow;

namespace {

/// Tile one linalg op with SCF loops and replace its results.
static LogicalResult tileOneOp(IRRewriter &rewriter, linalg::LinalgOp op,
                               ArrayRef<int64_t> tileSizes,
                               ArrayRef<int64_t> loopOrder) {
  auto tilingInterface = dyn_cast<TilingInterface>(op.getOperation());
  if (!tilingInterface)
    return failure();

  unsigned numLoops = op.getNumLoops();
  if (tileSizes.empty())
    return failure();
  SmallVector<int64_t> sizes(tileSizes.begin(),
                             tileSizes.begin() +
                                 std::min<size_t>(tileSizes.size(), numLoops));

  scf::SCFTilingOptions options;
  SmallVector<OpFoldResult> sizesOfr;
  for (int64_t size : sizes)
    sizesOfr.push_back(rewriter.getIndexAttr(size));
  options.setTileSizes(sizesOfr);
  if (loopOrder.size() == numLoops) {
    options.setInterchange(loopOrder);
  } else if (!loopOrder.empty() && loopOrder.size() == sizes.size()) {
    options.setInterchange(loopOrder);
  }

  FailureOr<scf::SCFTilingResult> tiled =
      scf::tileUsingSCF(rewriter, tilingInterface, options);
  if (failed(tiled))
    return failure();

  rewriter.replaceOp(op, tiled->replacements);
  return success();
}

static void collectScheduledOps(ModuleOp module,
                                SmallVectorImpl<linalg::LinalgOp> &ops) {
  module.walk([&](linalg::LinalgOp op) {
    if (getSchedule(op.getOperation()))
      ops.push_back(op);
  });
}

/// Fuse producers of extracted slices into the surrounding loop nest.
static unsigned fuseProducers(ModuleOp module, IRRewriter &rewriter) {
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp loop) { loops.push_back(loop); });

  unsigned fused = 0;
  // Innermost loops first.
  for (scf::ForOp loop : llvm::reverse(loops)) {
    if (!loop->getBlock())
      continue;

    SmallVector<LoopLikeOpInterface> nest;
    for (Operation *parent = loop.getOperation(); parent;
         parent = parent->getParentOp()) {
      if (auto loopLike = dyn_cast<LoopLikeOpInterface>(parent))
        nest.push_back(loopLike);
    }
    std::reverse(nest.begin(), nest.end());

    SmallVector<tensor::ExtractSliceOp> slices;
    loop.walk([&](tensor::ExtractSliceOp slice) {
      if (slice->getBlock())
        slices.push_back(slice);
    });

    for (tensor::ExtractSliceOp slice : slices) {
      if (!slice->getBlock())
        continue;
      Operation *producer = slice.getSource().getDefiningOp();
      if (!producer || !isa<TilingInterface>(producer))
        continue;
      if (loop->isProperAncestor(producer))
        continue;
      std::optional<scf::SCFFuseProducerOfSliceResult> result =
          scf::tileAndFuseProducerOfSlice(rewriter, slice, nest);
      if (!result)
        continue;
      ++fused;
      emitTileflowRemark("tileflow fused producer into tiled loop nest");
    }
  }

  // A fused producer keeps a slice of its full-size output init. When that
  // init is a tensor.empty, recreate the empty at tile size so no full-size
  // buffer is ever materialized.
  SmallVector<tensor::ExtractSliceOp> emptySlices;
  module.walk([&](tensor::ExtractSliceOp slice) {
    if (slice.getSource().getDefiningOp<tensor::EmptyOp>())
      emptySlices.push_back(slice);
  });
  for (tensor::ExtractSliceOp slice : emptySlices) {
    if (!slice->getBlock())
      continue;
    auto ty = cast<RankedTensorType>(slice.getType());
    rewriter.setInsertionPoint(slice);
    Value empty = tensor::EmptyOp::create(rewriter, slice.getLoc(),
                                          ty.getShape(), ty.getElementType());
    rewriter.replaceOp(slice, empty);
  }

  // Drop producers that lost all their uses.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    module.walk([&](Operation *op) {
      if (op == module.getOperation())
        return;
      if (isOpTriviallyDead(op))
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      rewriter.eraseOp(op);
      changed = true;
    }
  }
  return fused;
}

static MemRefType scratchpadType(MemRefType source, bool workgroup) {
  if (workgroup) {
    auto space = IntegerAttr::get(IntegerType::get(source.getContext(), 64), 3);
    return MemRefType::get(source.getShape(), source.getElementType(),
                           MemRefLayoutAttrInterface{}, space);
  }
  return MemRefType::get(source.getShape(), source.getElementType());
}

static Value createScratchpad(OpBuilder &builder, Location loc, MemRefType type,
                              bool workgroup) {
  if (workgroup)
    return gpu::AllocOp::create(builder, loc, type, Type(), ValueRange{},
                                ValueRange{}, ValueRange{}, false)
        .getMemref();
  return memref::AllocaOp::create(builder, loc, type).getResult();
}

static StringRef operandName(Operation *op, unsigned index) {
  static const char *matmulNames[] = {"A", "B", "C"};
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op) && index < 3)
    return matmulNames[index];
  if (isa<linalg::GenericOp>(op))
    return index == 0 ? "A" : (index == 1 ? "B" : "C");
  return "";
}

/// Promote operands of tiled linalg ops into a local scratchpad. Inputs are
/// copied in; destination operands are also copied out after the op.
static unsigned promoteOperands(ModuleOp module, IRRewriter &rewriter,
                                StringRef space,
                                ArrayRef<std::string> names) {
  bool workgroup = space == "workgroup" || space == "gpu" || space == "shared";
  unsigned promoted = 0;

  SmallVector<linalg::LinalgOp> ops;
  module.walk([&](linalg::LinalgOp op) { ops.push_back(op); });

  for (linalg::LinalgOp op : ops) {
    if (!op->getBlock())
      continue;

    // Snapshot operands because we will replace them.
    SmallVector<std::pair<unsigned, Value>> candidates;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      auto subview = operand.getDefiningOp<memref::SubViewOp>();
      if (!subview)
        continue;
      auto memrefTy = dyn_cast<MemRefType>(subview.getType());
      if (!memrefTy || !memrefTy.hasStaticShape())
        continue;
      StringRef name = operandName(op.getOperation(), index);
      if (!names.empty() &&
          llvm::find(names, name.str()) == names.end())
        continue;
      candidates.emplace_back(index, operand);
    }

    DenseSet<Value> initValues;
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(op.getOperation()))
      for (Value init : dps.getDpsInits())
        initValues.insert(init);

    for (auto [index, original] : candidates) {
      auto subview = original.getDefiningOp<memref::SubViewOp>();
      auto memrefTy = cast<MemRefType>(subview.getType());
      MemRefType promotedTy = scratchpadType(memrefTy, workgroup);

      rewriter.setInsertionPoint(op);
      Value scratch =
          createScratchpad(rewriter, op->getLoc(), promotedTy, workgroup);
      linalg::CopyOp::create(rewriter, op->getLoc(), ValueRange{original},
                             ValueRange{scratch});
      op->setOperand(index, scratch);

      if (initValues.contains(original)) {
        rewriter.setInsertionPointAfter(op);
        linalg::CopyOp::create(rewriter, op->getLoc(), ValueRange{scratch},
                               ValueRange{original});
      }
      ++promoted;
      emitTileflowRemark("tileflow promoted operand #" + llvm::Twine(index) +
                         " to " + (workgroup ? "workgroup" : "private") +
                         " scratchpad");
    }
  }
  return promoted;
}

//===----------------------------------------------------------------------===//
// Apply schedule
//===----------------------------------------------------------------------===//

struct TileFlowApplySchedulePass
    : public mlir::tileflow::impl::TileFlowApplyScheduleBase<TileFlowApplySchedulePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<linalg::LinalgOp> ops;
    collectScheduledOps(module, ops);
    SmallVector<std::string> promoteNames;
    for (linalg::LinalgOp op : ops) {
      if (auto schedule = getSchedule(op.getOperation()))
        for (StringRef name : schedule.getPromoteNames())
          if (llvm::find(promoteNames, name.str()) == promoteNames.end())
            promoteNames.push_back(name.str());
    }
    if (ops.empty()) {
      emitTileflowRemark("tileflow-apply-schedule found no scheduled ops");
      return;
    }

    for (linalg::LinalgOp op : ops) {
      if (!op->getBlock())
        continue;
      ScheduleAttr schedule = getSchedule(op.getOperation());
      if (!schedule)
        continue;

      SmallVector<int64_t> tiles;
      if (!tileSizes.empty())
        tiles.assign(tileSizes.begin(), tileSizes.end());
      else
        tiles.assign(schedule.getTileSizes().begin(),
                     schedule.getTileSizes().end());

      SmallVector<int64_t> order;
      if (!loopOrder.empty())
        order.assign(loopOrder.begin(), loopOrder.end());
      else
        order.assign(schedule.getLoopOrder().begin(),
                     schedule.getLoopOrder().end());

      if (failed(tileOneOp(rewriter, op, tiles, order))) {
        emitTileflowRemark("tileflow-apply-schedule skipped op: not tileable");
        continue;
      }
      emitTileflowRemark("tileflow applied schedule to a tiled loop nest");
    }

    stripSchedules(module.getOperation());

    if (fuse) {
      unsigned count = fuseProducers(module, rewriter);
      emitTileflowRemark("tileflow fused " + llvm::Twine(count) +
                         " producer(s)");
    }

    if (promote) {
      if (moduleHasTensors(module) && failed(bufferizeModule(module))) {
        module.emitError("tileflow-apply-schedule: bufferization failed");
        signalPassFailure();
        return;
      }
      unsigned count = promoteOperands(module, rewriter, space, promoteNames);
      emitTileflowRemark("tileflow promoted " + llvm::Twine(count) +
                         " operand(s)");
    }
  }
};

//===----------------------------------------------------------------------===//
// Standalone tiling pass
//===----------------------------------------------------------------------===//

struct TileFlowTileForLocalityPass
    : public mlir::tileflow::impl::TileFlowTileForLocalityBase<TileFlowTileForLocalityPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<linalg::LinalgOp> ops;
    if (!tileSizes.empty()) {
      module.walk([&](linalg::LinalgOp op) { ops.push_back(op); });
    } else {
      collectScheduledOps(module, ops);
    }

    for (linalg::LinalgOp op : ops) {
      if (!op->getBlock())
        continue;
      SmallVector<int64_t> tiles(tileSizes.begin(), tileSizes.end());
      SmallVector<int64_t> order(loopOrder.begin(), loopOrder.end());
      if (tiles.empty()) {
        if (auto schedule = getSchedule(op.getOperation())) {
          tiles.assign(schedule.getTileSizes().begin(),
                       schedule.getTileSizes().end());
          order.assign(schedule.getLoopOrder().begin(),
                       schedule.getLoopOrder().end());
        }
      }
      if (tiles.empty())
        continue;
      if (failed(tileOneOp(rewriter, op, tiles, order))) {
        emitTileflowRemark("tileflow-tile-for-locality skipped non-tileable op");
        continue;
      }
      emitTileflowRemark(
          "tileflow tiled op with [" +
          llvm::Twine(llvm::join(llvm::map_range(tiles,
                                                 [](int64_t v) {
                                                   return std::to_string(v);
                                                 }),
                                  ", ")) +
          "]");
    }
    // The schedule has been consumed by this point.
    stripSchedules(module.getOperation());
  }
};

//===----------------------------------------------------------------------===//
// Standalone fusion pass
//===----------------------------------------------------------------------===//

struct TileFlowFuseProducersPass
    : public mlir::tileflow::impl::TileFlowFuseProducersBase<TileFlowFuseProducersPass> {
  using Base::Base;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    unsigned count = fuseProducers(getOperation(), rewriter);
    emitTileflowRemark("tileflow fused " + llvm::Twine(count) +
                       " producer(s) into loop nests");
  }
};

//===----------------------------------------------------------------------===//
// Standalone promotion pass
//===----------------------------------------------------------------------===//

struct TileFlowPromoteToScratchpadPass
    : public mlir::tileflow::impl::TileFlowPromoteToScratchpadBase<
          TileFlowPromoteToScratchpadPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    if (moduleHasTensors(module) && failed(bufferizeModule(module))) {
      module.emitError("tileflow-promote-to-scratchpad: bufferization failed");
      signalPassFailure();
      return;
    }

    std::vector<std::string> selected(operands.begin(), operands.end());
    unsigned count = promoteOperands(module, rewriter, space, selected);
    emitTileflowRemark("tileflow promoted " + llvm::Twine(count) +
                       " operand(s)");
  }
};

} // namespace
