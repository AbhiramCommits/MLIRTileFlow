#include "TileFlow/Transforms/Passes.h"

namespace mlir {
namespace tileflow {

#define GEN_PASS_REGISTRATION
#include "TileFlow/Transforms/Passes.h.inc"

void registerAllTileFlowPasses() { registerTileFlowPasses(); }

} // namespace tileflow
} // namespace mlir
