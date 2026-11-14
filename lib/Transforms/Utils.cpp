#include "TileFlow/Transforms/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::tileflow;

llvm::cl::opt<bool>
    tileflow::tileflowRemarks("tileflow-remarks",
                              llvm::cl::desc("Emit a remark for every TileFlow transformation"),
                              llvm::cl::init(false));

ScheduleAttr tileflow::getSchedule(Operation *op) {
  return dyn_cast_or_null<ScheduleAttr>(op->getAttr(kScheduleAttrName));
}

void tileflow::emitTileflowRemark(const llvm::Twine &message) {
  if (tileflowRemarks)
    llvm::errs() << "remark: " << message << "\n";
}

void tileflow::attachSchedule(Operation *op, ScheduleAttr schedule) {
  if (schedule)
    op->setAttr(kScheduleAttrName, schedule);
}

void tileflow::stripSchedules(Operation *op) {
  op->walk([](Operation *nested) {
    if (nested->hasAttr(kScheduleAttrName))
      nested->removeAttr(kScheduleAttrName);
  });
}

bool tileflow::moduleHasTensors(ModuleOp module) {
  bool hasTensor = false;
  module.walk([&](Operation *op) {
    if (llvm::any_of(op->getOperandTypes(), [](Type type) { return isa<TensorType>(type); }) ||
        llvm::any_of(op->getResultTypes(), [](Type type) { return isa<TensorType>(type); })) {
      hasTensor = true;
      return WalkResult::interrupt();
    }
    if (auto func = dyn_cast<FunctionOpInterface>(op)) {
      for (Type type : func.getArgumentTypes())
        if (isa<TensorType>(type)) {
          hasTensor = true;
          return WalkResult::interrupt();
        }
      for (Type type : func.getResultTypes())
        if (isa<TensorType>(type)) {
          hasTensor = true;
          return WalkResult::interrupt();
        }
    }
    return WalkResult::advance();
  });
  return hasTensor;
}

LogicalResult tileflow::bufferizeModule(ModuleOp module) {
  bufferization::OneShotBufferizePassOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.functionBoundaryTypeConversion = bufferization::LayoutMapOption::IdentityLayoutMap;
  PassManager pm(module.getContext());
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  return pm.run(module);
}
