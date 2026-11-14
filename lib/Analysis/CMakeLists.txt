add_mlir_library(TileFlowAnalysis
  LocalityCostModel.cpp
  TileFlowLocalityReport.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TileFlow/Analysis

  DEPENDS
  TileFlowOpsIncGen
  TileFlowPassIncGen

  LINK_LIBS PUBLIC
  TileFlowDialect
  MLIRPass
  MLIRIR
  MLIRSupport
  LLVMSupport
  )
