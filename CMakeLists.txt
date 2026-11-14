cmake_minimum_required(VERSION 3.20)
project(MLIRTileFlow VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add_mlir_tool -> llvm_add_tool excludes tools from ALL unless this is ON.
set(LLVM_BUILD_TOOLS ON CACHE BOOL "" FORCE)

if(NOT DEFINED MLIR_DIR)
  message(FATAL_ERROR
    "MLIR_DIR is not set. Run scripts/bootstrap.sh and configure with "
    "-DMLIR_DIR=<prefix>/lib/cmake/mlir")
endif()

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(TableGen)
if(EXISTS "${PROJECT_SOURCE_DIR}/.venv/bin/python")
  set(Python3_EXECUTABLE "${PROJECT_SOURCE_DIR}/.venv/bin/python"
      CACHE FILEPATH "Python interpreter used for lit" FORCE)
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Homebrew's MLIRConfig may leave the tablegen executables as bare names.
if(NOT EXISTS "${MLIR_TABLEGEN_EXE}")
  set(MLIR_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/mlir-tblgen")
endif()
if(NOT EXISTS "${MLIR_PDLL_TABLEGEN_EXE}")
  set(MLIR_PDLL_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/mlir-pdll")
endif()
message(STATUS "Using mlir-tblgen: ${MLIR_TABLEGEN_EXE}")

# Locate lit for check-tileflow. The repository venv is preferred.
find_program(TILEFLOW_LIT_EXECUTABLE NAMES lit
  HINTS "${PROJECT_SOURCE_DIR}/.venv/bin" "${LLVM_TOOLS_BINARY_DIR}")
if(TILEFLOW_LIT_EXECUTABLE)
  set(LLVM_EXTERNAL_LIT "${TILEFLOW_LIT_EXECUTABLE}" CACHE STRING "" FORCE)
  message(STATUS "Using lit: ${LLVM_EXTERNAL_LIT}")
else()
  message(WARNING "lit not found; install with 'python3 -m pip install lit'")
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(MLIR_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)
add_definitions(${LLVM_DEFINITIONS})

if(NOT MSVC)
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# Link the MLIR component archives directly. Homebrew's monolithic libMLIR
# does not export the full MLIR API, and mixing the dylib with the component
# archives duplicates MLIR's TypeID state, which breaks type interfaces at
# runtime. Linking only the component archives keeps a single copy of every
# TypeID in the final executable on both Homebrew and apt.llvm.org installs.
get_property(TILEFLOW_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(TILEFLOW_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(MLIR_LINK_MLIR_DYLIB)
  foreach(component MLIRMlirOptMain MLIROptLib MLIRParser)
    find_library(TILEFLOW_${component}_LIB NAMES ${component}
      PATHS "${MLIR_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
    if(NOT TILEFLOW_${component}_LIB)
      message(FATAL_ERROR "${component} archive not found in ${MLIR_INSTALL_PREFIX}/lib")
    endif()
  endforeach()
  set(TILEFLOW_MLIR_TOOL_LIBS
    ${TILEFLOW_MLIRMlirOptMain_LIB}
    ${TILEFLOW_MLIROptLib_LIB}
    MLIRPluginsLib
    MLIRBytecodeWriter
    MLIRBytecodeReader
    MLIRRemarkStreamer
    MLIRDebug
    MLIRObservers
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    ${TILEFLOW_DIALECT_LIBS}
    ${TILEFLOW_CONVERSION_LIBS}
    ${TILEFLOW_MLIRParser_LIB})
else()
  set(TILEFLOW_MLIR_TOOL_LIBS
    MLIRMlirOptMain
    MLIROptLib
    MLIRPluginsLib
    MLIRBytecodeWriter
    MLIRBytecodeReader
    MLIRRemarkStreamer
    MLIRDebug
    MLIRObservers
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    ${TILEFLOW_DIALECT_LIBS}
    ${TILEFLOW_CONVERSION_LIBS}
    MLIRParser)
endif()

add_subdirectory(include/TileFlow)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

enable_testing()

# The full suite is driven by ctest: lit for the compiler tests and pytest for
# the numerical validation.
add_test(NAME check-tileflow
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-tileflow)
add_test(NAME pytest-tileflow
  COMMAND ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/python -q)
set_tests_properties(pytest-tileflow PROPERTIES
  ENVIRONMENT "TILEFLOW_OPT=$<TARGET_FILE:tileflow-opt>;TILEFLOW_TUNE=$<TARGET_FILE:tileflow-tune>;TILEFLOW_REPO=${PROJECT_SOURCE_DIR}")

