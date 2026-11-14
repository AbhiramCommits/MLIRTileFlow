#include "TileFlow/Analysis/LocalityCostModel.h"

#include "TileFlow/Dialect/TileFlowOps.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::tileflow;

MachineDescription MachineDescription::getDefault() {
  MachineDescription machine;
  machine.name = "Apple M3 (MacBook Air, arm64)";
  machine.levels = {
      {"L1D", 65536, 128},
      {"L2", 4194304, 128},
      {"DRAM", 17179869184LL, 128},
  };
  machine.scratchpadBytes = 65536;
  machine.vectorWidthBytes = 16;
  return machine;
}

FailureOr<MachineDescription> MachineDescription::loadFromFile(StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return failure();

  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed)
    return failure();

  llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return failure();

  MachineDescription machine;
  if (auto name = root->getString("name"))
    machine.name = name->str();
  if (auto scratchpad = root->getInteger("scratchpad_bytes"))
    machine.scratchpadBytes = *scratchpad;
  if (auto vectorWidth = root->getInteger("vector_width_bytes"))
    machine.vectorWidthBytes = *vectorWidth;

  llvm::json::Array *levels = root->getArray("levels");
  if (!levels)
    return failure();
  for (llvm::json::Value &entry : *levels) {
    llvm::json::Object *level = entry.getAsObject();
    if (!level)
      return failure();
    MachineLevel parsedLevel;
    if (auto name = level->getString("name"))
      parsedLevel.name = name->str();
    if (auto capacity = level->getInteger("capacity_bytes"))
      parsedLevel.capacityBytes = *capacity;
    if (auto line = level->getInteger("line_bytes"))
      parsedLevel.lineBytes = *line;
    machine.levels.push_back(parsedLevel);
  }
  if (machine.levels.size() < 2)
    return failure();
  return machine;
}

namespace {

static bool isPermutation(ArrayRef<int64_t> order) {
  llvm::SmallBitVector seen(order.size());
  for (int64_t value : order) {
    if (value < 0 || value >= static_cast<int64_t>(order.size()))
      return false;
    if (seen.test(value))
      return false;
    seen.set(value);
  }
  return true;
}

static int64_t ceilDiv(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }

static int64_t powerOfTwoBucket(int64_t bytes) {
  if (bytes <= 1)
    return 1;
  int64_t bucket = 1;
  while (bucket < bytes)
    bucket <<= 1;
  return bucket;
}

} // namespace

LocalityCostResult LocalityCostModel::evaluate(const LocalityCostInput &input,
                                               const ScheduleParams &schedule) const {
  LocalityCostResult result;
  const int64_t rank = static_cast<int64_t>(input.extents.size());

  if (static_cast<int64_t>(schedule.tiles.size()) != rank ||
      static_cast<int64_t>(schedule.order.size()) != rank) {
    result.feasible = false;
    result.reason = "schedule rank does not match the iteration space";
    return result;
  }
  if (!isPermutation(schedule.order)) {
    result.feasible = false;
    result.reason = "schedule loop order is not a permutation";
    return result;
  }

  SmallVector<int64_t> tiles(rank);
  SmallVector<int64_t> trips(rank);
  llvm::SmallVector<int64_t> positionOf(rank);
  int64_t totalTileIterations = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (schedule.tiles[i] <= 0) {
      result.feasible = false;
      result.reason = "tile sizes must be strictly positive";
      return result;
    }
    tiles[i] = std::max<int64_t>(1, std::min(schedule.tiles[i], input.extents[i]));
    trips[i] = ceilDiv(input.extents[i], tiles[i]);
    totalTileIterations *= trips[i];
  }
  for (int64_t position = 0; position < rank; ++position)
    positionOf[schedule.order[position]] = position;
  result.totalTileIterations = totalTileIterations;

  // Bytes of `operand` touched by one full execution of the loops at and
  // inside `startPosition`. Outer map dimensions contribute their tile size;
  // inner map dimensions contribute their full extent (all tiles are visited);
  // inner non-map dimensions contribute their trip count (reuse references).
  auto touchedBytes = [&](const OperandAccess &operand, int64_t startPosition) -> int64_t {
    int64_t bytes = operand.elementBytes;
    for (int64_t loop = 0; loop < rank; ++loop) {
      int64_t position = positionOf[loop];
      if (position <= startPosition) {
        if (operand.dependsOnLoop[loop])
          bytes *= tiles[loop];
      } else {
        bytes *= operand.dependsOnLoop[loop] ? input.extents[loop] : trips[loop];
      }
    }
    return bytes;
  };

  // Per-operand tile statistics.
  struct OperandStats {
    const OperandAccess *operand;
    int64_t tileBytes = 0;
    int64_t distinctTiles = 0;
  };
  SmallVector<OperandStats> stats;
  int64_t blockWorkingSet = 0;
  for (const OperandAccess &operand : input.operands) {
    OperandStats stat;
    stat.operand = &operand;
    int64_t bytes = operand.elementBytes;
    int64_t distinct = 1;
    for (int64_t loop = 0; loop < rank; ++loop) {
      if (!operand.dependsOnLoop[loop])
        continue;
      bytes *= tiles[loop];
      distinct *= trips[loop];
    }
    stat.tileBytes = bytes;
    stat.distinctTiles = distinct;
    blockWorkingSet += bytes;
    stats.push_back(stat);
  }

  // Reuse distance at each loop that repeats an operand's data.
  int64_t distanceSum = 0;
  int64_t distanceCount = 0;
  for (const OperandStats &stat : stats) {
    for (int64_t loop = 0; loop < rank; ++loop) {
      if (stat.operand->dependsOnLoop[loop])
        continue;
      int64_t position = positionOf[loop];
      int64_t distance = 0;
      for (const OperandStats &other : stats)
        distance += touchedBytes(*other.operand, position);
      result.reuseHistogram.push_back({powerOfTwoBucket(distance), 1});
      distanceSum += distance;
      ++distanceCount;
    }
  }
  result.averageReuseDistanceBytes =
      distanceCount ? static_cast<double>(distanceSum) / distanceCount : 0.0;

  // Merge histogram buckets.
  llvm::sort(result.reuseHistogram, [](const ReuseBucket &a, const ReuseBucket &b) {
    return a.bucketBytes < b.bucketBytes;
  });
  SmallVector<ReuseBucket> merged;
  for (const ReuseBucket &bucket : result.reuseHistogram) {
    if (!merged.empty() && merged.back().bucketBytes == bucket.bucketBytes)
      merged.back().count += bucket.count;
    else
      merged.push_back(bucket);
  }
  result.reuseHistogram = std::move(merged);

  int64_t totalElements = 1;
  for (int64_t extent : input.extents)
    totalElements *= extent;

  // Working-set-vs-capacity reasoning for each cache level. When the whole
  // innermost block fits, reuse loops multiply by their tile trip counts (the
  // block is revisited once per tile). When it does not fit, the same data is
  // re-streamed for every element of the reuse loop.
  unsigned numCaches = machine.getNumCaches();
  result.bytesPerLevel.assign(machine.levels.size(), 0);
  SmallVector<int64_t> cacheTraffic(numCaches, 0);
  for (unsigned cacheLevel = 0; cacheLevel < numCaches; ++cacheLevel) {
    int64_t capacity = machine.levels[cacheLevel].capacityBytes;
    bool blockFits = blockWorkingSet <= capacity;
    int64_t traffic = 0;
    for (const OperandStats &stat : stats) {
      int64_t loadFactor = 1;
      for (int64_t loop = 0; loop < rank; ++loop) {
        if (stat.operand->dependsOnLoop[loop])
          continue;
        loadFactor *= blockFits ? trips[loop] : input.extents[loop];
      }
      int64_t bytes = stat.distinctTiles * stat.tileBytes * loadFactor;
      int64_t upperBound = totalElements * stat.operand->elementBytes * 4; // QK+PV headroom
      bytes = std::min(bytes, upperBound);
      traffic += bytes;
    }
    cacheTraffic[cacheLevel] = traffic;
    result.bytesPerLevel[cacheLevel] = traffic;
  }
  result.dramBytes = numCaches ? cacheTraffic[numCaches - 1] : 0;
  if (result.bytesPerLevel.size() > numCaches)
    result.bytesPerLevel.back() = result.dramBytes;

  result.arithmeticIntensity =
      result.dramBytes ? static_cast<double>(input.flops) / static_cast<double>(result.dramBytes)
                       : 0.0;

  // Scratchpad footprint of the promoted operands.
  int64_t scratchpad = 0;
  for (const OperandStats &stat : stats) {
    if (llvm::find(schedule.promote, stat.operand->name) == schedule.promote.end())
      continue;
    scratchpad += stat.tileBytes;
  }
  result.scratchpadBytes = scratchpad;

  result.arithmeticIntensity =
      result.dramBytes ? static_cast<double>(input.flops) / static_cast<double>(result.dramBytes)
                       : 0.0;

  if (scratchpad > machine.scratchpadBytes) {
    result.feasible = false;
    result.reason = "promoted tile footprint exceeds scratchpad capacity";
  }
  return result;
}

FailureOr<LocalityCostInput> tileflow::buildCostInput(Operation *op) {
  LocalityCostInput input;
  auto elementBytes = [](Type elementType) -> int64_t {
    if (auto floatType = dyn_cast<FloatType>(elementType))
      return floatType.getWidth() / 8;
    return 4;
  };

  if (auto matmul = dyn_cast<MatmulOp>(op)) {
    input.opName = "tileflow.matmul";
    input.extents = matmul.getIterationSpaceExtents();
    auto ty = cast<RankedTensorType>(matmul.getA().getType());
    int64_t bytes = elementBytes(ty.getElementType());
    input.operands.push_back({"A", {true, false, true}, bytes});
    input.operands.push_back({"B", {false, true, true}, bytes});
    input.operands.push_back({"C", {true, true, false}, bytes});
    input.flops = 2 * input.extents[0] * input.extents[1] * input.extents[2];
    return input;
  }
  if (auto softmax = dyn_cast<SoftmaxRowOp>(op)) {
    input.opName = "tileflow.softmax_row";
    input.extents = softmax.getIterationSpaceExtents();
    auto ty = cast<RankedTensorType>(softmax.getInput().getType());
    int64_t bytes = elementBytes(ty.getElementType());
    input.operands.push_back({"X", {true, true}, bytes});
    input.operands.push_back({"Y", {true, true}, bytes});
    input.flops = 5 * input.extents[0] * input.extents[1];
    return input;
  }
  if (auto attention = dyn_cast<AttentionOp>(op)) {
    input.opName = "tileflow.attention";
    input.extents = attention.getIterationSpaceExtents();
    auto ty = cast<RankedTensorType>(attention.getQ().getType());
    int64_t bytes = elementBytes(ty.getElementType());
    input.operands.push_back({"Q", {true, true, false, true}, bytes});
    input.operands.push_back({"K", {true, false, true, true}, bytes});
    input.operands.push_back({"V", {true, false, true, true}, bytes});
    input.operands.push_back({"S", {true, true, true, false}, bytes});
    input.operands.push_back({"O", {true, true, false, true}, bytes});
    int64_t heads = input.extents[0], seqQ = input.extents[1];
    int64_t seqK = input.extents[2], headDim = input.extents[3];
    input.flops = 4 * heads * seqQ * seqK * headDim + 5 * heads * seqQ * seqK;
    return input;
  }
  return failure();
}
