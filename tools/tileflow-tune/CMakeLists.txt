add_mlir_tool(tileflow-tune tileflow-tune.cpp)
target_link_libraries(tileflow-tune PRIVATE LLVMSupport)
mlir_check_link_libraries(tileflow-tune)
