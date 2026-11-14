get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_tool(tileflow-opt tileflow-opt.cpp)

target_link_libraries(tileflow-opt PRIVATE
  TileFlowDialect
  ${dialect_libs}
  ${conversion_libs}
  MLIRMlirOptMain
  MLIROptLib
  MLIRParser
  MLIRSupport
  )

mlir_check_link_libraries(tileflow-opt)
