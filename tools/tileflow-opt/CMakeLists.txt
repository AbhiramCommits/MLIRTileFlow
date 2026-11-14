add_mlir_tool(tileflow-opt tileflow-opt.cpp)

target_link_libraries(tileflow-opt PRIVATE
  TileFlowDialect
  TileFlowTransforms
  ${TILEFLOW_MLIR_TOOL_LIBS}
  MLIRSupport
  )

mlir_check_link_libraries(tileflow-opt)
