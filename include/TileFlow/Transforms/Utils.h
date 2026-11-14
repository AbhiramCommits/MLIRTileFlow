#ifndef TILEFLOW_TRANSFORMS_UTILS_H
#define TILEFLOW_TRANSFORMS_UTILS_H

#include "TileFlow/Dialect/TileFlowAttrs.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace tileflow {

/// Discardable attribute name used to carry a schedule through the linalg and
/// loop stages.
constexpr StringLiteral kScheduleAttrName = "tileflow.schedule";
/// Enables per-transformation remarks (`-tileflow-remarks`).
extern llvm::cl::opt<bool> tileflowRemarks;

/// Print a transformation remark to stderr when `-tileflow-remarks` is set.
void emitTileflowRemark(const llvm::Twine &message);

/// Read the schedule attached to `op`, or null.
ScheduleAttr getSchedule(Operation *op);
/// Attach `schedule` to `op` as a discardable attribute.
void attachSchedule(Operation *op, ScheduleAttr schedule);
/// Remove the schedule attribute from `op` and every nested op.
void stripSchedules(Operation *op);

/// Return true when `module` still contains tensor-typed SSA values or tensor
/// function signatures.
bool moduleHasTensors(ModuleOp module);

/// Run one-shot bufferization with function boundaries bufferized to identity
/// layout memrefs. Returns failure when bufferization cannot prove safety.
LogicalResult bufferizeModule(ModuleOp module);

} // namespace tileflow
} // namespace mlir

#endif // TILEFLOW_TRANSFORMS_UTILS_H
