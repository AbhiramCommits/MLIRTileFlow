add_mlir_dialect_library(TileFlowDialect
  TileFlowDialect.cpp
  TileFlowAttrs.cpp
  TileFlowOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TileFlow/Dialect

  DEPENDS
  TileFlowOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRBytecodeOpInterface
  MLIRLinalgDialect
  MLIRTensorDialect
  )
