#include "TileFlow/Dialect/TileFlowAttrs.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"

#define GET_ATTRDEF_CLASSES
#include "TileFlow/Dialect/TileFlowAttrs.cpp.inc"

void mlir::tileflow::TileFlowDialect::initializeAttributes() {
  addAttributes<ScheduleAttr>();
}

using namespace mlir;
using namespace mlir::tileflow;

//===----------------------------------------------------------------------===//
// Construction helpers
//===----------------------------------------------------------------------===//

ScheduleAttr ScheduleAttr::get(MLIRContext *ctx, ArrayRef<int64_t> tiles,
                               ArrayRef<int64_t> order, int64_t unroll,
                               ArrayRef<StringRef> promote, StringRef target) {
  SmallVector<Attribute> promoteAttrs;
  promoteAttrs.reserve(promote.size());
  for (StringRef name : promote)
    promoteAttrs.push_back(StringAttr::get(ctx, name));
  return get(ctx, DenseI64ArrayAttr::get(ctx, tiles),
             DenseI64ArrayAttr::get(ctx, order), unroll,
             ArrayAttr::get(ctx, promoteAttrs), StringAttr::get(ctx, target));
}

SmallVector<StringRef> ScheduleAttr::getPromoteNames() const {
  SmallVector<StringRef> names;
  for (Attribute attr : getPromote()) {
    auto str = dyn_cast<StringAttr>(attr);
    if (str)
      names.push_back(str.getValue());
  }
  return names;
}

bool ScheduleAttr::promotes(StringRef name) const {
  for (Attribute attr : getPromote()) {
    auto str = dyn_cast<StringAttr>(attr);
    if (str && str.getValue() == name)
      return true;
  }
  return false;
}

bool ScheduleAttr::isIdentity(ArrayRef<int64_t> extents) const {
  ArrayRef<int64_t> tiles = getTileSizes();
  if (tiles.size() != extents.size() || getUnroll() != 1 ||
      !getPromote().empty())
    return false;
  for (auto [tile, extent] : llvm::zip_equal(tiles, extents))
    if (tile != extent)
      return false;
  return true;
}

ScheduleAttr ScheduleAttr::merge(ScheduleAttr outer, ScheduleAttr inner) {
  ArrayRef<int64_t> outerTiles = outer.getTileSizes();
  ArrayRef<int64_t> innerTiles = inner.getTileSizes();
  if (outerTiles.size() != innerTiles.size())
    return nullptr;

  SmallVector<int64_t> tiles;
  tiles.reserve(outerTiles.size());
  for (auto [o, i] : llvm::zip_equal(outerTiles, innerTiles))
    tiles.push_back(std::min(o, i));

  // The inner schedule is closest to the data; keep its loop order.
  SmallVector<int64_t> order(inner.getLoopOrder());

  SmallVector<StringRef> promote(outer.getPromoteNames());
  for (StringRef name : inner.getPromoteNames())
    if (llvm::find(promote, name) == promote.end())
      promote.push_back(name);

  int64_t unroll = outer.getUnroll() * inner.getUnroll();
  return ScheduleAttr::get(inner.getContext(), tiles, order, unroll, promote,
                           outer.getTarget().getValue());
}

//===----------------------------------------------------------------------===//
// Parsing and printing
//===----------------------------------------------------------------------===//

Attribute ScheduleAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  MLIRContext *ctx = parser.getContext();
  SmallVector<int64_t> tiles, order;
  SmallVector<std::string> promote;
  int64_t unroll = 1;
  std::string target = "cpu";
  bool sawTiles = false, sawOrder = false;

  auto parseI64List = [&](SmallVectorImpl<int64_t> &out) -> ParseResult {
    return parser.parseCommaSeparatedList(
        AsmParser::Delimiter::Square, [&]() -> ParseResult {
          int64_t value;
          if (parser.parseInteger(value))
            return failure();
          out.push_back(value);
          return success();
        });
  };

  do {
    StringRef key;
    if (parser.parseKeyword(&key) || parser.parseEqual())
      return {};

    if (key == "tiles") {
      if (parseI64List(tiles))
        return {};
      sawTiles = true;
    } else if (key == "order") {
      if (parseI64List(order))
        return {};
      sawOrder = true;
    } else if (key == "unroll") {
      if (parser.parseInteger(unroll))
        return {};
    } else if (key == "promote") {
      if (parser.parseCommaSeparatedList(
              AsmParser::Delimiter::Square, [&]() -> ParseResult {
                std::string name;
                if (parser.parseString(&name))
                  return failure();
                promote.push_back(name);
                return success();
              }))
        return {};
    } else if (key == "target") {
      std::string value;
      if (parser.parseString(&value))
        return {};
      target = value;
    } else {
      parser.emitError(parser.getCurrentLocation())
          << "unknown tileflow.schedule field '" << key
          << "'; expected tiles, order, unroll, promote or target";
      return {};
    }
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater())
    return {};

  if (!sawTiles || !sawOrder) {
    parser.emitError(parser.getCurrentLocation())
        << "tileflow.schedule requires both 'tiles' and 'order'";
    return {};
  }

  SmallVector<Attribute> promoteAttrs;
  for (const std::string &name : promote)
    promoteAttrs.push_back(StringAttr::get(ctx, name));

  return ScheduleAttr::get(ctx, DenseI64ArrayAttr::get(ctx, tiles),
                           DenseI64ArrayAttr::get(ctx, order), unroll,
                           ArrayAttr::get(ctx, promoteAttrs),
                           StringAttr::get(ctx, target));
}

void ScheduleAttr::print(AsmPrinter &printer) const {
  printer << "<tiles = [";
  llvm::interleaveComma(getTileSizes(), printer);
  printer << "], order = [";
  llvm::interleaveComma(getLoopOrder(), printer);
  printer << "], unroll = " << getUnroll() << ", promote = [";
  llvm::interleaveComma(getPromote(), printer, [&](Attribute attr) {
    printer << "\"" << cast<StringAttr>(attr).getValue() << "\"";
  });
  printer << "], target = \"" << getTarget().getValue() << "\">";
}

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

static LogicalResult verifyScheduleImpl(
    function_ref<InFlightDiagnostic()> emitError, DenseI64ArrayAttr tiles,
    DenseI64ArrayAttr order, int64_t unroll, ArrayAttr promote,
    StringAttr target) {
  if (!tiles || !order || !promote || !target)
    return emitError() << "tileflow.schedule is missing a required field";

  ArrayRef<int64_t> tileSizes = tiles.asArrayRef();
  ArrayRef<int64_t> loopOrder = order.asArrayRef();

  if (tileSizes.empty())
    return emitError() << "tileflow.schedule requires at least one tile size";

  if (tileSizes.size() != loopOrder.size())
    return emitError() << "tileflow.schedule 'tiles' has "
                       << tileSizes.size() << " entries but 'order' has "
                       << loopOrder.size()
                       << "; they must match the iteration-space rank";

  for (auto [index, tile] : llvm::enumerate(tileSizes)) {
    if (tile <= 0)
      return emitError() << "tileflow.schedule tile size at position " << index
                         << " must be strictly positive, got " << tile;
  }

  llvm::SmallBitVector seen(loopOrder.size());
  for (int64_t position : loopOrder) {
    if (position < 0 || position >= static_cast<int64_t>(loopOrder.size()))
      return emitError() << "tileflow.schedule order entry " << position
                         << " is out of range for rank " << loopOrder.size();
    if (seen.test(position))
      return emitError() << "tileflow.schedule order is not a permutation: "
                         << position << " appears more than once";
    seen.set(position);
  }

  for (Attribute attr : promote) {
    auto name = dyn_cast<StringAttr>(attr);
    if (!name)
      return emitError()
             << "tileflow.schedule promote entries must be strings";
    StringRef value = name.getValue();
    if (value != "A" && value != "B" && value != "Q" && value != "K" &&
        value != "V")
      return emitError() << "tileflow.schedule promotes unknown operand '"
                         << value
                         << "'; expected a subset of {\"A\", \"B\", \"Q\", "
                            "\"K\", \"V\"}";
  }

  if (unroll < 1)
    return emitError() << "tileflow.schedule unroll must be at least 1, got "
                       << unroll;

  StringRef targetValue = target.getValue();
  if (targetValue != "cpu" && targetValue != "gpu")
    return emitError() << "tileflow.schedule target must be \"cpu\" or "
                          "\"gpu\", got \""
                       << targetValue << "\"";

  return success();
}

LogicalResult ScheduleAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI64ArrayAttr tiles,
    DenseI64ArrayAttr order, int64_t unroll, ArrayAttr promote,
    StringAttr target) {
  return verifyScheduleImpl(emitError, tiles, order, unroll, promote, target);
}

LogicalResult ScheduleAttr::verifySchedule(
    function_ref<InFlightDiagnostic()> emitError) const {
  return verifyScheduleImpl(emitError, getTiles(), getOrder(), getUnroll(),
                            getPromote(), getTarget());
}
