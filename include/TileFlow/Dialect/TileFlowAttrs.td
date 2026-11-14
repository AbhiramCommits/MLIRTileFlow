#ifndef TILEFLOW_ATTRS_TD
#define TILEFLOW_ATTRS_TD

include "TileFlowDialect.td"
include "mlir/IR/OpBase.td"

// A tiling schedule attached to a tileflow operation.
//
//   #tileflow.schedule<tiles = [32, 32, 32], order = [0, 1, 2], unroll = 1,
//                       promote = ["A", "B"], target = "cpu">
//
// `tiles` and `order` have one entry per loop of the operation's iteration
// space. `order` is a permutation describing the loop nest order after
// interchange. `promote` lists the operand names that must be staged through
// a scratchpad. `target` selects the CPU or GPU lowering path.
def TileFlow_ScheduleAttr : AttrDef<TileFlow_Dialect, "Schedule"> {
  let mnemonic = "schedule";
  let summary = "Tiling, loop order and scratchpad promotion schedule";
  let description = [{
    The schedule is deliberately a first-class attribute so that it can be
    produced by `tileflow-tune`, written back into the IR and consumed by
    `-tileflow-apply-schedule`.
  }];

  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$tiles,
    "::mlir::DenseI64ArrayAttr":$order,
    "int64_t":$unroll,
    "::mlir::ArrayAttr":$promote,
    "::mlir::StringAttr":$target
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static ScheduleAttr get(::mlir::MLIRContext *ctx,
                            ::llvm::ArrayRef<int64_t> tiles,
                            ::llvm::ArrayRef<int64_t> order, int64_t unroll,
                            ::llvm::ArrayRef<::llvm::StringRef> promote,
                            ::llvm::StringRef target);

    ::llvm::ArrayRef<int64_t> getTileSizes() const {
      return getTiles().asArrayRef();
    }
    ::llvm::ArrayRef<int64_t> getLoopOrder() const {
      return getOrder().asArrayRef();
    }
    ::llvm::SmallVector<::llvm::StringRef> getPromoteNames() const;
    bool promotes(::llvm::StringRef name) const;
    bool isCpu() const { return getTarget().getValue() == "cpu"; }

    /// True when `tiles` covers `extents` exactly, there is no unrolling and
    /// no promotion: the schedule is a no-op for a shape with these extents.
    bool isIdentity(::llvm::ArrayRef<int64_t> extents) const;

    /// Validate every schedule field. Used by op verifiers so diagnostics are
    /// attached to the operation that carries the schedule.
    ::llvm::LogicalResult verifySchedule(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const;

    /// Compose an outer and an inner schedule for nested schedule regions.
    /// The inner schedule runs closest to the data, so its order wins and the
    /// tile sizes are the element-wise minimum. Unroll factors multiply and
    /// promotion sets are unioned. Returns null on incompatible ranks.
    static ScheduleAttr merge(ScheduleAttr outer, ScheduleAttr inner);
  }];
}

#endif // TILEFLOW_ATTRS_TD
