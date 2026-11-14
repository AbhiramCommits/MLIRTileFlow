#include "TileFlow/Transforms/Passes.h"
#include "TileFlow/Transforms/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace tileflow {
#define GEN_PASS_DEF_TILEFLOWTOLOOPS
#include "TileFlow/Transforms/Passes.h.inc"
} // namespace tileflow
} // namespace mlir

using namespace mlir;
using namespace mlir::tileflow;

namespace {

struct TileFlowToLoopsPass
    : public mlir::tileflow::impl::TileFlowToLoopsBase<TileFlowToLoopsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    if (moduleHasTensors(module) && failed(bufferizeModule(module))) {
      module.emitError("tileflow-to-loops: one-shot bufferization failed");
      signalPassFailure();
      return;
    }

    // Bufferization can leave a self-copy for in-place loop results. CSE first
    // so identical subviews become one value, then drop the copy. Removing it
    // also keeps the final LLVM module free of the memrefCopy runtime symbol.
    PassManager cleanup(ctx);
    cleanup.addPass(createCSEPass());
    if (failed(cleanup.run(module))) {
      signalPassFailure();
      return;
    }
    {
      IRRewriter rewriter(ctx);
      SmallVector<memref::CopyOp> selfCopies;
      module.walk([&](memref::CopyOp copy) {
        if (copy.getSource() == copy.getTarget())
          selfCopies.push_back(copy);
      });
      for (memref::CopyOp copy : selfCopies)
        rewriter.eraseOp(copy);
    }

    if (vectorize) {
      IRRewriter rewriter(ctx);
      SmallVector<linalg::LinalgOp> ops;
      module.walk([&](linalg::LinalgOp op) { ops.push_back(op); });
      for (linalg::LinalgOp op : ops) {
        if (!op->getBlock())
          continue;
        FailureOr<linalg::VectorizationResult> result =
            linalg::vectorize(rewriter, op.getOperation());
        if (failed(result))
          continue;
        rewriter.replaceOp(op, result->replacements);
      }
    }

    PassManager pm(ctx);
    pm.addPass(createConvertLinalgToLoopsPass());
    if (failed(pm.run(module))) {
      signalPassFailure();
      return;
    }

    if (unroll > 1) {
      SmallVector<scf::ForOp> innermost;
      module.walk([&](scf::ForOp loop) {
        bool hasNestedLoop = false;
        loop.getBody()->walk([&](scf::ForOp nested) {
          if (nested != loop)
            hasNestedLoop = true;
        });
        if (!hasNestedLoop)
          innermost.push_back(loop);
      });
      for (scf::ForOp loop : innermost) {
        if (loop->getBlock())
          (void)loopUnrollByFactor(loop, unroll);
      }
    }
  }
};

} // namespace
