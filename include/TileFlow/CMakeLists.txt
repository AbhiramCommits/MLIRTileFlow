file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Dialect)

set(LLVM_TARGET_DEFINITIONS Dialect/TileFlowOps.td)
mlir_tablegen(Dialect/TileFlowOps.h.inc -gen-op-decls)
mlir_tablegen(Dialect/TileFlowOps.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect/TileFlowOpsDialect.h.inc -gen-dialect-decls
  -dialect=tileflow)
mlir_tablegen(Dialect/TileFlowOpsDialect.cpp.inc -gen-dialect-defs
  -dialect=tileflow)
add_public_tablegen_target(TileFlowOpsIncGen)
