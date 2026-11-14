#ifndef TILEFLOW_ANALYSIS_LOCALITYCOSTMODEL_H
#define TILEFLOW_ANALYSIS_LOCALITYCOSTMODEL_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace tileflow {

/// One level of the memory hierarchy, outermost last (DRAM).
struct MachineLevel {
  std::string name;
  int64_t capacityBytes = 0;
  int64_t lineBytes = 64;
};

struct MachineDescription {
  std::string name;
  std::vector<MachineLevel> levels; // L1, L2, ..., DRAM
  int64_t scratchpadBytes = 0;
  int64_t vectorWidthBytes = 16;

  /// Number of levels that are caches (all but DRAM).
  unsigned getNumCaches() const {
    return levels.empty() ? 0 : static_cast<unsigned>(levels.size() - 1);
  }
  int64_t getDramCapacity() const { return levels.empty() ? 0 : levels.back().capacityBytes; }

  /// Real parameters of the machine this build was measured on.
  static MachineDescription getDefault();
  /// Load a machine description from JSON.
  static FailureOr<MachineDescription> loadFromFile(llvm::StringRef path);
};

/// Description of how one operand is indexed by the iteration space.
struct OperandAccess {
  std::string name;
  /// One entry per iteration-space loop: true when the operand depends on it.
  llvm::SmallVector<bool> dependsOnLoop;
  int64_t elementBytes = 4;
};

/// Everything the model needs about a compute operation.
struct LocalityCostInput {
  std::string opName;
  llvm::SmallVector<int64_t> extents;
  llvm::SmallVector<OperandAccess> operands;
  int64_t flops = 0;
};

struct ScheduleParams {
  llvm::SmallVector<int64_t> tiles;
  llvm::SmallVector<int64_t> order;
  int64_t unroll = 1;
  llvm::SmallVector<std::string> promote;
  std::string target = "cpu";
};

struct ReuseBucket {
  int64_t bucketBytes = 0;
  int64_t count = 0;
};

struct LocalityCostResult {
  bool feasible = true;
  std::string reason;
  /// Bytes moved into each hierarchy level, indexed like machine.levels.
  /// The last entry is the estimated DRAM traffic.
  llvm::SmallVector<int64_t> bytesPerLevel;
  int64_t dramBytes = 0;
  double arithmeticIntensity = 0.0;
  double averageReuseDistanceBytes = 0.0;
  llvm::SmallVector<ReuseBucket> reuseHistogram;
  int64_t scratchpadBytes = 0;
  int64_t totalTileIterations = 0;
};

class LocalityCostModel {
public:
  explicit LocalityCostModel(MachineDescription machine) : machine(std::move(machine)) {}

  const MachineDescription &getMachine() const { return machine; }

  LocalityCostResult evaluate(const LocalityCostInput &input, const ScheduleParams &schedule) const;

private:
  MachineDescription machine;
};

/// Build the cost-model input for a tileflow op (matmul, softmax_row or
/// attention). Returns failure for other operations.
FailureOr<LocalityCostInput> buildCostInput(Operation *op);

} // namespace tileflow
} // namespace mlir

#endif // TILEFLOW_ANALYSIS_LOCALITYCOSTMODEL_H
