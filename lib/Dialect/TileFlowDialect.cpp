#include "TileFlow/Dialect/TileFlowDialect.h"

#include "TileFlow/Dialect/TileFlowAttrs.h"
#include "TileFlow/Dialect/TileFlowOps.h"

#include "TileFlow/Dialect/TileFlowOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::tileflow;

void TileFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TileFlow/Dialect/TileFlowOps.cpp.inc"
      >();
  initializeAttributes();
}
