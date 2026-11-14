#include "TileFlow/Transforms/Passes.h"
#include "TileFlow/Transforms/Utils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace tileflow {
#define GEN_PASS_DEF_TILEFLOWLOWER
#define GEN_PASS_DEF_TILEFLOWLOWERTOGPU
#include "TileFlow/Transforms/Passes.h.inc"
} // namespace tileflow
} // namespace mlir

using namespace mlir;
using namespace mlir::tileflow;

namespace {

static bool isValidStop(StringRef value, ArrayRef<StringRef> allowed) {
  return llvm::is_contained(allowed, value);
}

//===----------------------------------------------------------------------===//
// Composite CPU pipeline
//===----------------------------------------------------------------------===//

struct TileFlowLowerPass : public mlir::tileflow::impl::TileFlowLowerBase<TileFlowLowerPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    StringRef stop = stopAfter;
    if (!isValidStop(stop, {"linalg", "schedule", "loops", "llvm"})) {
      module.emitError() << "invalid stop-after value '" << stop
                         << "'; expected linalg, schedule, loops or llvm";
      signalPassFailure();
      return;
    }

    PassManager pm(&getContext());
    pm.addPass(createTileFlowToLinalg());
    if (stop == "linalg") {
      if (failed(pm.run(module)))
        signalPassFailure();
      return;
    }

    TileFlowApplyScheduleOptions scheduleOptions;
    scheduleOptions.fuse = true;
    scheduleOptions.promote = true;
    pm.addPass(createTileFlowApplySchedule(scheduleOptions));
    if (stop == "schedule") {
      if (failed(pm.run(module)))
        signalPassFailure();
      return;
    }

    pm.addPass(createTileFlowToLoops());
    if (stop == "loops") {
      if (failed(pm.run(module)))
        signalPassFailure();
      return;
    }

    pm.addPass(createTileFlowToLLVM());
    if (failed(pm.run(module)))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// GPU pipeline: scf.forall -> gpu -> nvvm
//===----------------------------------------------------------------------===//

static void tileScheduledOpsForGPU(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  SmallVector<linalg::LinalgOp> ops;
  module.walk([&](linalg::LinalgOp op) {
    if (getSchedule(op.getOperation()))
      ops.push_back(op);
  });

  for (linalg::LinalgOp op : ops) {
    if (!op->getBlock())
      continue;
    ScheduleAttr schedule = getSchedule(op.getOperation());
    if (!schedule)
      continue;

    unsigned numLoops = op.getNumLoops();
    ArrayRef<int64_t> scheduleTiles = schedule.getTileSizes();
    SmallVector<utils::IteratorType> iteratorTypes =
        op.getIteratorTypesArray();
    SmallVector<OpFoldResult> sizes;
    SmallVector<Attribute> mapping;
    MLIRContext *ctx = module.getContext();
    for (unsigned i = 0; i < numLoops && i < scheduleTiles.size(); ++i) {
      // Reduction loops must stay sequential: distributing them over threads
      // would create cross-thread races.
      bool isReduction = i < iteratorTypes.size() &&
                         iteratorTypes[i] == utils::IteratorType::reduction;
      if (isReduction) {
        sizes.push_back(rewriter.getIndexAttr(0));
        continue;
      }
      sizes.push_back(rewriter.getIndexAttr(scheduleTiles[i]));
      if (mapping.empty())
        mapping.push_back(gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimX));
      else
        mapping.push_back(gpu::GPUThreadMappingAttr::get(
            ctx, static_cast<gpu::MappingId>((mapping.size() - 1) % 3)));
    }

    scf::SCFTilingOptions options;
    options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    options.setTileSizes(sizes);
    options.setMapping(mapping);

    auto tiled = scf::tileUsingSCF(rewriter, cast<TilingInterface>(op.getOperation()),
                                   options);
    if (failed(tiled))
      continue;
    rewriter.replaceOp(op, tiled->replacements);
  }
  stripSchedules(module.getOperation());
}

struct TileFlowLowerToGPUPass
    : public mlir::tileflow::impl::TileFlowLowerToGPUBase<TileFlowLowerToGPUPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    StringRef stop = stopAfter;
    if (!isValidStop(stop, {"forall", "gpu", "nvvm"})) {
      module.emitError() << "invalid stop-after value '" << stop
                         << "'; expected forall, gpu or nvvm";
      signalPassFailure();
      return;
    }

    PassManager pm(&getContext());
    pm.addPass(createTileFlowToLinalg());
    if (failed(pm.run(module))) {
      signalPassFailure();
      return;
    }

    tileScheduledOpsForGPU(module);
    if (stop == "forall")
      return;

    if (moduleHasTensors(module) && failed(bufferizeModule(module))) {
      module.emitError("tileflow-lower-to-gpu: bufferization failed");
      signalPassFailure();
      return;
    }

    PassManager loops(&getContext());
    loops.addPass(createConvertLinalgToLoopsPass());
    loops.addPass(createForallToParallelLoopPass());
    if (failed(loops.run(module))) {
      signalPassFailure();
      return;
    }

    // Lower scf.parallel to gpu.launch. The parallel loop was produced from
    // an scf.forall with GPU mappings, so one loop iteration maps to one GPU
    // block; the body is moved into the launch region.
    SmallVector<scf::ParallelOp> parallels;
    module.walk([&](scf::ParallelOp parallel) {
      if (parallel->getNumResults() == 0)
        parallels.push_back(parallel);
    });
    MLIRContext *ctx = &getContext();
    {
      IRRewriter rewriter(ctx);
      for (scf::ParallelOp parallel : parallels) {
        if (!parallel->getBlock())
          continue;
        Location loc = parallel.getLoc();
        rewriter.setInsertionPoint(parallel);
        Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
        SmallVector<Value> trips;
        for (auto [lb, ub, step] : llvm::zip_equal(
                 parallel.getLowerBound(), parallel.getUpperBound(),
                 parallel.getStep())) {
          Value range = arith::SubIOp::create(rewriter, loc, ub, lb);
          Value rounded = arith::AddIOp::create(
              rewriter, loc, range,
              arith::SubIOp::create(rewriter, loc, step, one));
          trips.push_back(arith::DivUIOp::create(rewriter, loc, rounded, step));
        }
        while (trips.size() < 3)
          trips.push_back(one);

        auto launch = gpu::LaunchOp::create(rewriter, loc, trips[0], trips[1],
                                            trips[2], one, one, one);
        Block &launchBody = launch.getBody().front();
        SmallVector<Value> ivs;
        {
          OpBuilder builder(&launchBody, launchBody.begin());
          for (auto [index, lb, step] : llvm::enumerate(
                   parallel.getLowerBound(), parallel.getStep())) {
            Value blockId = launchBody.getArgument(index);
            Value scaled = builder.create<arith::MulIOp>(loc, blockId, step);
            ivs.push_back(builder.create<arith::AddIOp>(loc, scaled, lb));
          }
        }

        OpBuilder terminatorBuilder(&launchBody, launchBody.end());
        auto terminator = terminatorBuilder.create<gpu::TerminatorOp>(loc);

        Block &body = *parallel.getBody();
        for (auto [argument, iv] : llvm::zip_equal(body.getArguments(), ivs))
          argument.replaceAllUsesWith(iv);
        for (Operation &op : llvm::make_early_inc_range(body.getOperations())) {
          if (op.hasTrait<OpTrait::IsTerminator>())
            continue;
          op.moveBefore(terminator);
        }
        parallel.erase();
      }
    }
    if (stop == "gpu")
      return;

    PassManager nvvm(&getContext());
    nvvm.addPass(createGpuKernelOutliningPass());
    nvvm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps());
    if (failed(nvvm.run(module)))
      signalPassFailure();
  }
};

} // namespace
