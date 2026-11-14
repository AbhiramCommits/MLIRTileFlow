add_mlir_library(TileFlowTransforms
  ApplySchedule.cpp
  Passes.cpp
  Pipelines.cpp
  TileFlowToLinalg.cpp
  ToLLVM.cpp
  ToLoops.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TileFlow/Transforms

  DEPENDS
  TileFlowOpsIncGen
  TileFlowPassIncGen

  LINK_LIBS PUBLIC
  TileFlowDialect
  MLIRPass
  MLIRTransforms
  MLIRBufferizationTransforms
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRMemRefTransforms
  MLIRGPUTransforms
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFToControlFlow
  MLIRVectorToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIRMathToLLVM
  MLIRIndexToLLVM
  MLIRControlFlowToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRAffineToStandard
  MLIRSCFToGPU
  MLIRGPUToNVVMTransforms
  MLIRNVVMToLLVM
  )
