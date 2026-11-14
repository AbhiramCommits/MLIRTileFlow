#ifndef TILEFLOW_ATTRS_H
#define TILEFLOW_ATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "TileFlow/Dialect/TileFlowDialect.h"

#define GET_ATTRDEF_CLASSES
#include "TileFlow/Dialect/TileFlowAttrs.h.inc"

#endif // TILEFLOW_ATTRS_H
