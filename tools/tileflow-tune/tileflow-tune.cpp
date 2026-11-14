#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::OptionCategory tuneCategory("tileflow-tune options");

static llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                            llvm::cl::desc("<input .mlir>"),
                                            llvm::cl::init("-"),
                                            llvm::cl::cat(tuneCategory));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "tileflow-tune: data-locality cost model schedule selector\n");

  llvm::errs() << "tileflow-tune: schedule selection is implemented in a later "
                  "stage (input: "
               << inputFile << ")\n";
  return 0;
}
