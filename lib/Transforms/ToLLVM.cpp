#include "TileFlow/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace tileflow {
#define GEN_PASS_DEF_TILEFLOWTOLLVM
#include "TileFlow/Transforms/Passes.h.inc"
} // namespace tileflow
} // namespace mlir

using namespace mlir;
using namespace mlir::tileflow;

namespace {

struct TileFlowToLLVMPass : public mlir::tileflow::impl::TileFlowToLLVMBase<TileFlowToLLVMPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(&getContext());

    pm.addPass(createLowerAffinePass());
    pm.addPass(createConvertVectorToSCFPass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createUBToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    if (failed(pm.run(module)))
      signalPassFailure();
  }
};

} // namespace
