#ifndef TILEFLOW_OPS_H
#define TILEFLOW_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TileFlow/Dialect/TileFlowAttrs.h"
#include "TileFlow/Dialect/TileFlowDialect.h"

#define GET_OP_CLASSES
#include "TileFlow/Dialect/TileFlowOps.h.inc"

#endif // TILEFLOW_OPS_H
