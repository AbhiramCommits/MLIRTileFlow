set(TILEFLOW_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TILEFLOW_BINARY_DIR ${PROJECT_BINARY_DIR})
set(TILEFLOW_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(TILEFLOW_TEST_DEPENDS
  tileflow-opt
  tileflow-tune
  FileCheck
  count
  not
  )

add_lit_testsuite(check-tileflow "Running the TileFlow regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TILEFLOW_TEST_DEPENDS}
  )

set_target_properties(check-tileflow PROPERTIES FOLDER "Tests")

add_lit_testsuites(TILEFLOW ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${TILEFLOW_TEST_DEPENDS}
  )
