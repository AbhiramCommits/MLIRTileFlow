#ifndef TILEFLOW_DIALECT_TD
#define TILEFLOW_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TileFlow_Dialect : Dialect {
  let name = "tileflow";
  let cppNamespace = "::mlir::tileflow";
  let summary = "Tiled tensor and attention operations with explicit schedules";
  let description = [{
    The TileFlow dialect models tensor computations whose data-locality
    strategy is carried explicitly as a `#tileflow.schedule` attribute.
    It is the entry point of a staged lowering pipeline:

      tileflow -> linalg -> scheduled loops -> affine/scf/vector -> LLVM

    Operations are pure and operate on ranked tensors of f32 or f16 with
    static shapes. The schedule attribute is consumed by the scheduling
    passes and discarded once it has been applied.
  }];
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    /// Defined in TileFlowAttrs.cpp where generated attribute storage is
    /// complete.
    void initializeAttributes();
  }];
}

class TileFlow_Op<string mnemonic, list<Trait> traits = []>
    : Op<TileFlow_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

#endif // TILEFLOW_DIALECT_TD
