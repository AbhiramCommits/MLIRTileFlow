add_mlir_tool(tileflow-tune tileflow-tune.cpp)
target_link_libraries(tileflow-tune PRIVATE
  TileFlowAnalysis
  TileFlowDialect
  ${TILEFLOW_MLIR_TOOL_LIBS}
  MLIRSupport
  )
mlir_check_link_libraries(tileflow-tune)
