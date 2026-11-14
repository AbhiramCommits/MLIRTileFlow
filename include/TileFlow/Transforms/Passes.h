#ifndef TILEFLOW_TRANSFORMS_PASSES_H
#define TILEFLOW_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tileflow {

#define GEN_PASS_DECL
#include "TileFlow/Transforms/Passes.h.inc"

/// Register every TileFlow pass with the global registry.
void registerAllTileFlowPasses();

} // namespace tileflow
} // namespace mlir

#endif // TILEFLOW_TRANSFORMS_PASSES_H
