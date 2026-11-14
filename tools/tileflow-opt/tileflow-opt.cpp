#include "TileFlow/Dialect/TileFlowDialect.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::tileflow::TileFlowDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TileFlow optimizer driver\n", registry));
}
