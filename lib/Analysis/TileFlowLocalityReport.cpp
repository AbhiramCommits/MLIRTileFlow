#include "TileFlow/Analysis/LocalityCostModel.h"
#include "TileFlow/Transforms/Passes.h"
#include "TileFlow/Dialect/TileFlowOps.h"
#include "TileFlow/Transforms/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tileflow {
#define GEN_PASS_DEF_TILEFLOWLOCALITYREPORT
#include "TileFlow/Transforms/Passes.h.inc"
} // namespace tileflow
} // namespace mlir

using namespace mlir;
using namespace mlir::tileflow;

namespace {

static llvm::json::Value resultToJSON(const LocalityCostInput &input,
                                      const LocalityCostResult &result, ScheduleAttr schedule) {
  llvm::json::Object object;
  object["op"] = input.opName;
  llvm::json::Array tiles;
  for (int64_t tile : schedule.getTileSizes())
    tiles.push_back(tile);
  object["tiles"] = std::move(tiles);
  llvm::json::Array order;
  for (int64_t value : schedule.getLoopOrder())
    order.push_back(value);
  object["order"] = std::move(order);
  llvm::json::Array promote;
  for (StringRef name : schedule.getPromoteNames())
    promote.push_back(name.str());
  object["promote"] = std::move(promote);
  object["feasible"] = result.feasible;
  if (!result.feasible)
    object["reason"] = result.reason;
  llvm::json::Array bytesPerLevel;
  for (int64_t bytes : result.bytesPerLevel)
    bytesPerLevel.push_back(bytes);
  object["bytes_per_level"] = std::move(bytesPerLevel);
  object["dram_bytes"] = result.dramBytes;
  object["arithmetic_intensity"] = result.arithmeticIntensity;
  object["average_reuse_distance_bytes"] = result.averageReuseDistanceBytes;
  llvm::json::Array histogram;
  for (const ReuseBucket &bucket : result.reuseHistogram) {
    llvm::json::Object entry;
    entry["bucket_bytes"] = bucket.bucketBytes;
    entry["count"] = bucket.count;
    histogram.push_back(std::move(entry));
  }
  object["reuse_distance_histogram"] = std::move(histogram);
  object["scratchpad_bytes"] = result.scratchpadBytes;
  object["flops"] = input.flops;
  object["tile_iterations"] = result.totalTileIterations;
  return llvm::json::Value(std::move(object));
}

struct TileFlowLocalityReportPass
    : public mlir::tileflow::impl::TileFlowLocalityReportBase<TileFlowLocalityReportPass> {
  using Base::Base;

  void runOnOperation() override {
    MachineDescription machine = MachineDescription::getDefault();
    if (!machinePath.empty()) {
      auto loaded = MachineDescription::loadFromFile(machinePath);
      if (failed(loaded)) {
        getOperation().emitError() << "cannot load machine description '" << machinePath << "'";
        signalPassFailure();
        return;
      }
      machine = std::move(*loaded);
    }

    LocalityCostModel model(machine);
    llvm::json::Array ops;
    getOperation().walk([&](Operation *op) {
      ScheduleAttr schedule;
      if (auto matmul = dyn_cast<MatmulOp>(op))
        schedule = matmul.getScheduleAttr();
      else if (auto softmax = dyn_cast<SoftmaxRowOp>(op))
        schedule = softmax.getScheduleAttr();
      else if (auto attention = dyn_cast<AttentionOp>(op))
        schedule = attention.getScheduleAttr();
      if (!schedule)
        schedule = getSchedule(op);
      if (!schedule)
        return;
      auto input = buildCostInput(op);
      if (failed(input))
        return;
      ScheduleParams params;
      for (int64_t tile : schedule.getTileSizes())
        params.tiles.push_back(tile);
      for (int64_t value : schedule.getLoopOrder())
        params.order.push_back(value);
      params.unroll = schedule.getUnroll();
      for (StringRef name : schedule.getPromoteNames())
        params.promote.push_back(name.str());
      params.target = schedule.getTarget().getValue().str();
      LocalityCostResult result = model.evaluate(*input, params);
      ops.push_back(resultToJSON(*input, result, schedule));
    });

    llvm::json::Object root;
    root["machine"] = machine.name;
    llvm::json::Array levels;
    for (const MachineLevel &level : machine.levels) {
      llvm::json::Object entry;
      entry["name"] = level.name;
      entry["capacity_bytes"] = level.capacityBytes;
      entry["line_bytes"] = level.lineBytes;
      levels.push_back(std::move(entry));
    }
    root["levels"] = std::move(levels);
    root["scratchpad_bytes"] = machine.scratchpadBytes;
    root["ops"] = std::move(ops);
    llvm::outs() << llvm::json::Value(std::move(root)) << "\n";
  }
};

} // namespace
