#include "TileFlow/Analysis/LocalityCostModel.h"
#include "TileFlow/Dialect/TileFlowDialect.h"
#include "TileFlow/Dialect/TileFlowOps.h"
#include "TileFlow/Transforms/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::tileflow;

static llvm::cl::OptionCategory tuneCategory("tileflow-tune options");

static llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                            llvm::cl::desc("<input .mlir>"),
                                            llvm::cl::init(""),
                                            llvm::cl::cat(tuneCategory));
static llvm::cl::opt<std::string> machinePath(
    "machine", llvm::cl::desc("Machine description JSON"),
    llvm::cl::init("bench/machine.json"), llvm::cl::cat(tuneCategory));
static llvm::cl::opt<std::string> jsonPath(
    "json", llvm::cl::desc("Ranking JSON output"),
    llvm::cl::init("results/cost_model_ranking.json"),
    llvm::cl::cat(tuneCategory));
static llvm::cl::opt<std::string> logPath(
    "log", llvm::cl::desc("Tuning log output"),
    llvm::cl::init("results/tune_log.txt"), llvm::cl::cat(tuneCategory));
static llvm::cl::opt<std::string> annotateInput(
    "annotate", llvm::cl::desc("Write the winning schedule into this .mlir"),
    llvm::cl::init(""), llvm::cl::cat(tuneCategory));
static llvm::cl::opt<std::string> annotateOutput(
    "o", llvm::cl::desc("Output file for --annotate"),
    llvm::cl::init(""), llvm::cl::cat(tuneCategory));
static llvm::cl::opt<unsigned> topCount(
    "top", llvm::cl::desc("Number of ranked candidates to print"),
    llvm::cl::init(20), llvm::cl::cat(tuneCategory));

namespace {

struct Candidate {
  ScheduleParams schedule;
  LocalityCostResult result;
};

static std::vector<std::vector<int64_t>>
permutations(int64_t rank) {
  std::vector<std::vector<int64_t>> result;
  std::vector<int64_t> current;
  std::vector<bool> used(rank, false);
  std::function<void()> rec = [&]() {
    if (static_cast<int64_t>(current.size()) == rank) {
      result.push_back(current);
      return;
    }
    for (int64_t i = 0; i < rank; ++i) {
      if (used[i])
        continue;
      used[i] = true;
      current.push_back(i);
      rec();
      current.pop_back();
      used[i] = false;
    }
  };
  rec();
  return result;
}

static std::vector<int64_t> tileChoices(ArrayRef<int64_t> extents) {
  int64_t maxExtent = *std::max_element(extents.begin(), extents.end());
  std::vector<int64_t> choices;
  for (int64_t value = 8; value <= maxExtent && value <= 128; value *= 2)
    choices.push_back(value);
  if (choices.empty())
    choices.push_back(std::max<int64_t>(1, maxExtent));
  if (llvm::find(choices, maxExtent) == choices.end())
    choices.push_back(maxExtent);
  return choices;
}

static std::vector<std::string> promoteSetFor(StringRef opName) {
  if (opName == "tileflow.matmul")
    return {"A", "B"};
  if (opName == "tileflow.attention")
    return {"K", "V"};
  return {};
}

static bool better(const Candidate &a, const Candidate &b) {
  if (a.result.feasible != b.result.feasible)
    return a.result.feasible;
  if (a.result.dramBytes != b.result.dramBytes)
    return a.result.dramBytes < b.result.dramBytes;
  // Break DRAM ties by innermost-level traffic: a schedule that keeps the
  // working set in L1 moves less data even when DRAM traffic is equal.
  if (!a.result.bytesPerLevel.empty() && !b.result.bytesPerLevel.empty() &&
      a.result.bytesPerLevel.front() != b.result.bytesPerLevel.front())
    return a.result.bytesPerLevel.front() < b.result.bytesPerLevel.front();
  return a.result.arithmeticIntensity > b.result.arithmeticIntensity;
}

static FailureOr<Operation *> findTunableOp(ModuleOp module) {
  Operation *found = nullptr;
  module.walk([&](Operation *op) {
    if (!found && succeeded(buildCostInput(op)))
      found = op;
  });
  if (!found)
    return failure();
  return found;
}

static void writeJSON(StringRef path, const MachineDescription &machine,
                      const LocalityCostInput &input,
                      ArrayRef<Candidate> candidates, size_t enumerated,
                      size_t feasible, const Candidate &winner) {
  llvm::SmallString<128> parent(path);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty())
    llvm::sys::fs::create_directories(parent);

  llvm::json::Object root;
  root["machine"] = machine.name;
  root["op"] = input.opName;
  llvm::json::Array extents;
  for (int64_t extent : input.extents)
    extents.push_back(extent);
  root["extents"] = std::move(extents);
  root["candidates_enumerated"] = static_cast<int64_t>(enumerated);
  root["candidates_ranked"] = static_cast<int64_t>(enumerated);
  root["candidates_feasible"] = static_cast<int64_t>(feasible);

  auto writeSchedule = [](const ScheduleParams &schedule) {
    llvm::json::Object object;
    llvm::json::Array tiles;
    for (int64_t tile : schedule.tiles)
      tiles.push_back(tile);
    object["tiles"] = std::move(tiles);
    llvm::json::Array order;
    for (int64_t value : schedule.order)
      order.push_back(value);
    object["order"] = std::move(order);
    object["unroll"] = schedule.unroll;
    llvm::json::Array promote;
    for (const std::string &name : schedule.promote)
      promote.push_back(name);
    object["promote"] = std::move(promote);
    object["target"] = schedule.target;
    return object;
  };

  llvm::json::Array ranked;
  for (size_t i = 0; i < candidates.size() && i < topCount; ++i) {
    const Candidate &candidate = candidates[i];
    llvm::json::Object object = writeSchedule(candidate.schedule);
    object["rank"] = static_cast<int64_t>(i + 1);
    object["feasible"] = candidate.result.feasible;
    if (!candidate.result.feasible)
      object["reason"] = candidate.result.reason;
    object["dram_bytes"] = candidate.result.dramBytes;
    llvm::json::Array bytesPerLevel;
    for (int64_t bytes : candidate.result.bytesPerLevel)
      bytesPerLevel.push_back(bytes);
    object["bytes_per_level"] = std::move(bytesPerLevel);
    object["arithmetic_intensity"] =
        candidate.result.arithmeticIntensity;
    object["average_reuse_distance_bytes"] =
        candidate.result.averageReuseDistanceBytes;
    object["scratchpad_bytes"] = candidate.result.scratchpadBytes;
    ranked.push_back(std::move(object));
  }
  root["ranking"] = std::move(ranked);
  root["winner"] = writeSchedule(winner.schedule);

  std::error_code error;
  llvm::raw_fd_ostream out(path, error);
  if (error) {
    llvm::errs() << "tileflow-tune: cannot write " << path << ": "
                 << error.message() << "\n";
    return;
  }
  out << llvm::json::Value(std::move(root)) << "\n";
}

static void writeLog(StringRef path, const MachineDescription &machine,
                     const LocalityCostInput &input, size_t enumerated,
                     size_t feasible, size_t bruteForce,
                     const Candidate &winner) {
  llvm::SmallString<128> parent(path);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty())
    llvm::sys::fs::create_directories(parent);

  std::error_code error;
  llvm::raw_fd_ostream out(path, error);
  if (error)
    return;
  out << "tileflow-tune log\n";
  out << "machine: " << machine.name << "\n";
  out << "op: " << input.opName << "\n";
  out << "candidates enumerated: " << enumerated << "\n";
  out << "candidates ranked by cost model: " << enumerated << "\n";
  out << "candidates feasible after scratchpad check: " << feasible << "\n";
  out << "candidates needing compilation under brute force: " << bruteForce
      << "\n";
  out << "winner tiles: [";
  llvm::interleaveComma(winner.schedule.tiles, out);
  out << "]\nwinner order: [";
  llvm::interleaveComma(winner.schedule.order, out);
  out << "]\nwinner promote: [";
  llvm::interleaveComma(winner.schedule.promote, out);
  out << "]\n";
  out << "winner estimated DRAM bytes: " << winner.result.dramBytes << "\n";
  out << "winner arithmetic intensity: " << winner.result.arithmeticIntensity
      << "\n";
  out << "winner average reuse distance bytes: "
      << winner.result.averageReuseDistanceBytes << "\n";
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "tileflow-tune: analytical data-locality schedule selector\n");

  std::string source = !annotateInput.empty() ? annotateInput : inputFile;
  if (source.empty()) {
    llvm::errs() << "tileflow-tune: provide an input .mlir file\n";
    return 1;
  }

  MachineDescription machine = MachineDescription::getDefault();
  auto loaded = MachineDescription::loadFromFile(machinePath);
  if (succeeded(loaded))
    machine = std::move(*loaded);
  else {
    llvm::errs() << "tileflow-tune: cannot load machine description '"
                 << machinePath << "'\n";
    return 1;
  }

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<TileFlowDialect>();

  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  auto module = parseSourceFile<ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "tileflow-tune: failed to parse " << source << "\n";
    return 1;
  }

  auto target = findTunableOp(*module);
  if (failed(target)) {
    llvm::errs() << "tileflow-tune: no tileflow compute op found in "
                 << source << "\n";
    return 1;
  }
  auto input = buildCostInput(*target);
  if (failed(input)) {
    llvm::errs() << "tileflow-tune: unsupported op\n";
    return 1;
  }

  LocalityCostModel model(machine);
  std::vector<int64_t> choices = tileChoices(input->extents);
  std::vector<std::vector<int64_t>> orders =
      permutations(input->extents.size());
  std::vector<std::string> promoteNames = promoteSetFor(input->opName);

  size_t enumerated = 0;
  std::vector<Candidate> candidates;
  std::vector<int64_t> tiles(input->extents.size());
  std::function<void(size_t)> enumerate = [&](size_t depth) {
    if (depth == tiles.size()) {
      for (const std::vector<int64_t> &order : orders) {
        for (int promote = 0; promote < 2; ++promote) {
          ScheduleParams schedule;
          schedule.tiles.append(tiles.begin(), tiles.end());
          schedule.order.append(order.begin(), order.end());
          if (promote && !promoteNames.empty())
            schedule.promote.assign(promoteNames.begin(), promoteNames.end());
          ++enumerated;
          Candidate candidate;
          candidate.schedule = schedule;
          candidate.result = model.evaluate(*input, schedule);
          candidates.push_back(std::move(candidate));
        }
      }
      return;
    }
    for (int64_t choice : choices) {
      // Do not tile a dimension by more than its extent: a larger tile only
      // creates dynamic boundary shapes.
      tiles[depth] = std::min(choice, input->extents[depth]);
      enumerate(depth + 1);
    }
  };
  enumerate(0);

  std::stable_sort(candidates.begin(), candidates.end(), better);
  size_t feasible = 0;
  for (const Candidate &candidate : candidates)
    if (candidate.result.feasible)
      ++feasible;

  Candidate winner = candidates.front();
  for (const Candidate &candidate : candidates) {
    if (candidate.result.feasible) {
      winner = candidate;
      break;
    }
  }

  size_t shown = std::min<size_t>(topCount, candidates.size());
  llvm::outs() << "# machine: " << machine.name << "\n";
  llvm::outs() << "# op: " << input->opName << " extents [";
  llvm::interleaveComma(input->extents, llvm::outs());
  llvm::outs() << "]\n";
  llvm::outs() << "# rank feasible dram_bytes intensity "
                  "avg_reuse_distance tiles order promote\n";
  for (size_t i = 0; i < shown; ++i) {
    const Candidate &candidate = candidates[i];
    llvm::outs() << i + 1 << " " << (candidate.result.feasible ? "yes" : "no ")
                 << " " << candidate.result.dramBytes << " "
                 << candidate.result.arithmeticIntensity << " "
                 << candidate.result.averageReuseDistanceBytes << " [";
    llvm::interleaveComma(candidate.schedule.tiles, llvm::outs());
    llvm::outs() << "] [";
    llvm::interleaveComma(candidate.schedule.order, llvm::outs());
    llvm::outs() << "] [";
    llvm::interleaveComma(candidate.schedule.promote, llvm::outs());
    llvm::outs() << "]";
    if (!candidate.result.feasible)
      llvm::outs() << "  # " << candidate.result.reason;
    llvm::outs() << "\n";
  }
  llvm::outs() << "# candidates enumerated: " << enumerated << "\n";
  llvm::outs() << "# candidates ranked by cost model: " << enumerated << "\n";
  llvm::outs() << "# candidates feasible after scratchpad check: " << feasible
               << "\n";
  llvm::outs() << "# candidates needing compilation under brute force: "
               << enumerated << "\n";

  writeJSON(jsonPath, machine, *input, candidates, enumerated, feasible,
            winner);
  writeLog(logPath, machine, *input, enumerated, feasible, enumerated, winner);

  if (!annotateInput.empty()) {
    if (annotateOutput.empty()) {
      llvm::errs() << "tileflow-tune: --annotate requires -o <output>\n";
      return 1;
    }
    SmallVector<StringRef> promoteNames;
    for (const std::string &name : winner.schedule.promote)
      promoteNames.push_back(name);
    ScheduleAttr schedule = ScheduleAttr::get(
        &context, winner.schedule.tiles, winner.schedule.order,
        winner.schedule.unroll, promoteNames, winner.schedule.target);
    Operation *op = *target;
    if (auto matmul = dyn_cast<MatmulOp>(op))
      matmul.setScheduleAttr(schedule);
    else if (auto softmax = dyn_cast<SoftmaxRowOp>(op))
      softmax.setScheduleAttr(schedule);
    else if (auto attention = dyn_cast<AttentionOp>(op))
      attention.setScheduleAttr(schedule);
    else
      op->setAttr("tileflow.schedule", schedule);

    std::error_code error;
    llvm::raw_fd_ostream out(annotateOutput, error);
    if (error) {
      llvm::errs() << "tileflow-tune: cannot write " << annotateOutput << "\n";
      return 1;
    }
    module->print(out);
  }
  return 0;
}
