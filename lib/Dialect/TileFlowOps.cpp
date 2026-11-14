#include "TileFlow/Dialect/TileFlowOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#define GET_OP_CLASSES
#include "TileFlow/Dialect/TileFlowOps.cpp.inc"

using namespace mlir;
using namespace mlir::tileflow;

static std::string formatShape(ArrayRef<int64_t> shape) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << "[";
  llvm::interleaveComma(shape, os);
  os << "]";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Iteration spaces
//===----------------------------------------------------------------------===//

SmallVector<int64_t> MatmulOp::getIterationSpaceExtents() {
  auto aTy = cast<RankedTensorType>(getA().getType());
  auto bTy = cast<RankedTensorType>(getB().getType());
  int64_t m = getTransposeA() ? aTy.getDimSize(1) : aTy.getDimSize(0);
  int64_t k = getTransposeA() ? aTy.getDimSize(0) : aTy.getDimSize(1);
  int64_t n = getTransposeB() ? bTy.getDimSize(0) : bTy.getDimSize(1);
  return {m, n, k};
}

SmallVector<int64_t> SoftmaxRowOp::getIterationSpaceExtents() {
  auto ty = cast<RankedTensorType>(getInput().getType());
  return {ty.getDimSize(0), ty.getDimSize(1)};
}

SmallVector<int64_t> AttentionOp::getIterationSpaceExtents() {
  auto qTy = cast<RankedTensorType>(getQ().getType());
  auto kTy = cast<RankedTensorType>(getK().getType());
  return {qTy.getDimSize(0), qTy.getDimSize(1), kTy.getDimSize(1), qTy.getDimSize(2)};
}

//===----------------------------------------------------------------------===//
// Custom assembly formats
//===----------------------------------------------------------------------===//

ParseResult MatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand a, b;
  Type aTy, bTy, outTy;
  NamedAttrList attrs;

  if (parser.parseOperand(a) || parser.parseComma() || parser.parseOperand(b))
    return failure();
  if (parser.parseOptionalAttrDict(attrs))
    return failure();
  if (parser.parseColonType(aTy) || parser.parseComma() || parser.parseType(bTy) ||
      parser.parseArrow() || parser.parseType(outTy))
    return failure();

  if (parser.resolveOperand(a, aTy, result.operands) ||
      parser.resolveOperand(b, bTy, result.operands))
    return failure();
  result.addTypes(outTy);

  auto &props = result.getOrAddProperties<MatmulOp::Properties>();
  if (Attribute attr = attrs.get("transpose_a")) {
    auto unit = dyn_cast<UnitAttr>(attr);
    if (!unit)
      return parser.emitError(parser.getCurrentLocation())
             << "expected unit attribute for 'transpose_a'";
    props.setTransposeA(unit);
  }
  if (Attribute attr = attrs.get("transpose_b")) {
    auto unit = dyn_cast<UnitAttr>(attr);
    if (!unit)
      return parser.emitError(parser.getCurrentLocation())
             << "expected unit attribute for 'transpose_b'";
    props.setTransposeB(unit);
  }
  if (Attribute attr = attrs.get("schedule")) {
    auto schedule = dyn_cast<ScheduleAttr>(attr);
    if (!schedule)
      return parser.emitError(parser.getCurrentLocation())
             << "expected a #tileflow.schedule attribute";
    props.setSchedule(schedule);
  }
  for (NamedAttribute attr : attrs)
    if (attr.getName() != "transpose_a" && attr.getName() != "transpose_b" &&
        attr.getName() != "schedule")
      result.addAttribute(attr.getName(), attr.getValue());
  return success();
}

void MatmulOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getA() << ", " << getB();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getA().getType() << ", " << getB().getType() << " -> "
          << getOutput().getType();
}

ParseResult SoftmaxRowOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  Type inputTy;
  NamedAttrList attrs;

  if (parser.parseOperand(input))
    return failure();
  if (parser.parseOptionalAttrDict(attrs))
    return failure();
  if (parser.parseColonType(inputTy))
    return failure();
  if (parser.resolveOperand(input, inputTy, result.operands))
    return failure();
  result.addTypes(inputTy);

  auto &props = result.getOrAddProperties<SoftmaxRowOp::Properties>();
  if (Attribute attr = attrs.get("axis")) {
    auto axis = dyn_cast<IntegerAttr>(attr);
    if (!axis || !axis.getType().isInteger(64))
      return parser.emitError(parser.getCurrentLocation())
             << "expected a 64-bit integer 'axis' attribute";
    props.setAxis(axis);
  }
  if (Attribute attr = attrs.get("schedule")) {
    auto schedule = dyn_cast<ScheduleAttr>(attr);
    if (!schedule)
      return parser.emitError(parser.getCurrentLocation())
             << "expected a #tileflow.schedule attribute";
    props.setSchedule(schedule);
  }
  for (NamedAttribute attr : attrs)
    if (attr.getName() != "axis" && attr.getName() != "schedule")
      result.addAttribute(attr.getName(), attr.getValue());
  return success();
}

void SoftmaxRowOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInput();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getInput().getType();
}

ParseResult AttentionOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand q, k, v;
  Type qTy, kTy, vTy, outTy;
  NamedAttrList attrs;

  if (parser.parseOperand(q) || parser.parseComma() || parser.parseOperand(k) ||
      parser.parseComma() || parser.parseOperand(v))
    return failure();
  if (parser.parseOptionalAttrDict(attrs))
    return failure();
  if (parser.parseColonType(qTy) || parser.parseComma() || parser.parseType(kTy) ||
      parser.parseComma() || parser.parseType(vTy) || parser.parseArrow() ||
      parser.parseType(outTy))
    return failure();

  if (parser.resolveOperand(q, qTy, result.operands) ||
      parser.resolveOperand(k, kTy, result.operands) ||
      parser.resolveOperand(v, vTy, result.operands))
    return failure();
  result.addTypes(outTy);

  auto &props = result.getOrAddProperties<AttentionOp::Properties>();
  if (Attribute attr = attrs.get("scale")) {
    auto scale = dyn_cast<FloatAttr>(attr);
    if (!scale)
      return parser.emitError(parser.getCurrentLocation())
             << "expected a floating-point 'scale' attribute";
    props.setScale(scale);
  }
  if (Attribute attr = attrs.get("causal")) {
    auto causal = dyn_cast<UnitAttr>(attr);
    if (!causal)
      return parser.emitError(parser.getCurrentLocation())
             << "expected unit attribute for 'causal'";
    props.setCausal(causal);
  }
  if (Attribute attr = attrs.get("schedule")) {
    auto schedule = dyn_cast<ScheduleAttr>(attr);
    if (!schedule)
      return parser.emitError(parser.getCurrentLocation())
             << "expected a #tileflow.schedule attribute";
    props.setSchedule(schedule);
  }
  for (NamedAttribute attr : attrs)
    if (attr.getName() != "scale" && attr.getName() != "causal" && attr.getName() != "schedule")
      result.addAttribute(attr.getName(), attr.getValue());
  return success();
}

void AttentionOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getQ() << ", " << getK() << ", " << getV();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getQ().getType() << ", " << getK().getType() << ", " << getV().getType()
          << " -> " << getOutput().getType();
}

ParseResult ScheduleRegionOp::parse(OpAsmParser &parser, OperationState &result) {
  Attribute schedule;
  if (parser.parseAttribute(schedule))
    return failure();
  if (!isa<ScheduleAttr>(schedule))
    return parser.emitError(parser.getCurrentLocation())
           << "expected a #tileflow.schedule attribute";

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{},
                         /*enableNameShadowing=*/false))
    return failure();
  if (parser.parseColonTypeList(result.types))
    return failure();

  result.getOrAddProperties<ScheduleRegionOp::Properties>().setSchedule(
      cast<ScheduleAttr>(schedule));
  return success();
}

void ScheduleRegionOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getSchedule() << ' ';
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer << " : ";
  llvm::interleaveComma(getResultTypes(), printer);
}

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  auto aTy = dyn_cast<RankedTensorType>(getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(getB().getType());
  auto outTy = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!aTy || !bTy || !outTy)
    return emitOpError() << "expects ranked tensor operands and result";
  if (aTy.getRank() != 2 || bTy.getRank() != 2 || outTy.getRank() != 2)
    return emitOpError() << "expects rank-2 operands and result, got ranks (" << aTy.getRank()
                         << ", " << bTy.getRank() << ", " << outTy.getRank() << ")";
  if (!aTy.hasStaticShape() || !bTy.hasStaticShape() || !outTy.hasStaticShape())
    return emitOpError() << "requires static operand and result shapes";
  if (aTy.getElementType() != bTy.getElementType() ||
      aTy.getElementType() != outTy.getElementType())
    return emitOpError() << "operands and result must have the same element type, got "
                         << aTy.getElementType() << ", " << bTy.getElementType() << ", "
                         << outTy.getElementType();

  int64_t m = getTransposeA() ? aTy.getDimSize(1) : aTy.getDimSize(0);
  int64_t k = getTransposeA() ? aTy.getDimSize(0) : aTy.getDimSize(1);
  int64_t kb = getTransposeB() ? bTy.getDimSize(1) : bTy.getDimSize(0);
  int64_t n = getTransposeB() ? bTy.getDimSize(0) : bTy.getDimSize(1);

  if (k != kb)
    return emitOpError() << "K dimension mismatch: A has K=" << k << " but B has K=" << kb;

  if (outTy.getDimSize(0) != m || outTy.getDimSize(1) != n)
    return emitOpError() << "result shape " << formatShape(outTy.getShape())
                         << " does not match the expected " << m << "x" << n;

  if (auto schedule = getScheduleAttr()) {
    if (failed(schedule.verifySchedule([&]() { return emitOpError(); })))
      return failure();
    int64_t rank = MatmulOp::kIterationRank;
    if (static_cast<int64_t>(schedule.getTileSizes().size()) != rank)
      return emitOpError() << "schedule has " << schedule.getTileSizes().size()
                           << " tile sizes but matmul iteration space (M, N, "
                              "K) has rank "
                           << rank;
  }
  return success();
}

LogicalResult SoftmaxRowOp::verify() {
  auto ty = dyn_cast<RankedTensorType>(getInput().getType());
  if (!ty)
    return emitOpError() << "expects a ranked tensor operand";
  if (ty.getRank() != 2)
    return emitOpError() << "expects a rank-2 tensor, got rank " << ty.getRank();
  if (!ty.hasStaticShape())
    return emitOpError() << "requires a static input shape";
  if (!ty.getElementType().isF32() && !ty.getElementType().isF16())
    return emitOpError() << "expects f32 or f16 elements, got " << ty.getElementType();

  int64_t axis = getAxisValue();
  if (axis < 0 || axis >= 2)
    return emitOpError() << "axis must be 0 or 1 for a rank-2 input, got " << axis;

  if (auto schedule = getScheduleAttr()) {
    if (failed(schedule.verifySchedule([&]() { return emitOpError(); })))
      return failure();
    if (static_cast<int64_t>(schedule.getTileSizes().size()) != SoftmaxRowOp::kIterationRank)
      return emitOpError() << "schedule has " << schedule.getTileSizes().size()
                           << " tile sizes but softmax iteration space (rows, "
                              "columns) has rank "
                           << SoftmaxRowOp::kIterationRank;
  }
  return success();
}

LogicalResult AttentionOp::verify() {
  auto qTy = dyn_cast<RankedTensorType>(getQ().getType());
  auto kTy = dyn_cast<RankedTensorType>(getK().getType());
  auto vTy = dyn_cast<RankedTensorType>(getV().getType());
  auto outTy = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!qTy || !kTy || !vTy || !outTy)
    return emitOpError() << "expects ranked tensor operands and result";
  if (qTy.getRank() != 3 || kTy.getRank() != 3 || vTy.getRank() != 3 || outTy.getRank() != 3)
    return emitOpError() << "expects rank-3 [heads, seq, head_dim] tensors, got ranks ("
                         << qTy.getRank() << ", " << kTy.getRank() << ", " << vTy.getRank() << ", "
                         << outTy.getRank() << ")";
  if (!qTy.hasStaticShape() || !kTy.hasStaticShape() || !vTy.hasStaticShape() ||
      !outTy.hasStaticShape())
    return emitOpError() << "requires static operand and result shapes";
  if (qTy.getElementType() != kTy.getElementType() ||
      qTy.getElementType() != vTy.getElementType() ||
      qTy.getElementType() != outTy.getElementType())
    return emitOpError() << "Q, K, V and result must have the same element "
                            "type";

  int64_t heads = qTy.getDimSize(0);
  if (kTy.getDimSize(0) != heads)
    return emitOpError() << "attention head count mismatch: Q has " << heads << " heads but K has "
                         << kTy.getDimSize(0);
  if (vTy.getDimSize(0) != heads)
    return emitOpError() << "attention head count mismatch: Q has " << heads << " heads but V has "
                         << vTy.getDimSize(0);

  int64_t headDim = qTy.getDimSize(2);
  if (kTy.getDimSize(2) != headDim)
    return emitOpError() << "attention head dimension mismatch: Q has " << headDim << " but K has "
                         << kTy.getDimSize(2);

  int64_t seqK = kTy.getDimSize(1);
  if (vTy.getDimSize(1) != seqK)
    return emitOpError() << "attention key sequence length mismatch: K has " << seqK
                         << " keys but V has " << vTy.getDimSize(1);

  int64_t seqQ = qTy.getDimSize(1);
  if (getCausalFlag() && seqQ != seqK)
    return emitOpError() << "causal attention requires seq_q == seq_k, got seq_q=" << seqQ
                         << " and seq_k=" << seqK;

  if (outTy.getDimSize(0) != heads || outTy.getDimSize(1) != seqQ ||
      outTy.getDimSize(2) != vTy.getDimSize(2))
    return emitOpError() << "result shape " << formatShape(outTy.getShape())
                         << " does not match [heads=" << heads << ", seq_q=" << seqQ
                         << ", v_dim=" << vTy.getDimSize(2) << "]";

  if (auto schedule = getScheduleAttr()) {
    if (failed(schedule.verifySchedule([&]() { return emitOpError(); })))
      return failure();
    if (static_cast<int64_t>(schedule.getTileSizes().size()) != AttentionOp::kIterationRank)
      return emitOpError() << "schedule has " << schedule.getTileSizes().size()
                           << " tile sizes but attention iteration space "
                              "(heads, seq_q, seq_k, head_dim) has rank "
                           << AttentionOp::kIterationRank;
  }
  return success();
}

LogicalResult ScheduleRegionOp::verify() {
  if (getBody().empty())
    return emitOpError() << "requires a non-empty body region";
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError() << "requires a single-block body";
  if (getNumResults() == 0)
    return emitOpError() << "requires at least one tensor result";

  Block &block = getBody().front();
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError() << "body must end with tileflow.yield";
  if (yield.getValues().size() != getNumResults())
    return emitOpError() << "yields " << yield.getValues().size() << " values but the region has "
                         << getNumResults() << " results";

  int64_t rank = -1;
  for (auto [index, pair] : llvm::enumerate(llvm::zip_equal(yield.getValues(), getResults()))) {
    Value yielded = std::get<0>(pair);
    Value result = std::get<1>(pair);
    if (yielded.getType() != result.getType())
      return emitOpError() << "yielded value #" << index << " has type " << yielded.getType()
                           << " but result #" << index << " has type " << result.getType();
    auto ty = dyn_cast<RankedTensorType>(result.getType());
    if (!ty)
      return emitOpError() << "result #" << index << " must be a ranked tensor";
    if (rank == -1)
      rank = ty.getRank();
    else if (rank != ty.getRank())
      return emitOpError() << "all results must have the same rank";
  }

  if (failed(getSchedule().verifySchedule([&]() { return emitOpError(); })))
    return failure();
  if (static_cast<int64_t>(getSchedule().getTileSizes().size()) != rank)
    return emitOpError() << "schedule has " << getSchedule().getTileSizes().size()
                         << " tile sizes but the region results have rank " << rank;
  return success();
}

//===----------------------------------------------------------------------===//
// Folding and canonicalization
//===----------------------------------------------------------------------===//

OpFoldResult MatmulOp::fold(FoldAdaptor adaptor) {
  auto outTy = getOutput().getType();
  // A matmul with a zero-sized dimension has an empty result. Materialize it
  // as an empty dense constant so the rest of the pipeline can DCE it.
  if (outTy.getNumElements() == 0)
    return DenseElementsAttr::get(outTy, ArrayRef<Attribute>{});
  return {};
}

namespace {

/// Fold `linalg.transpose` producers of a matmul into the matmul's
/// `transpose_a` / `transpose_b` flags. Only the [1, 0] permutation is
/// supported, which is exactly the 2-D transpose a matmul can absorb.
struct FoldTransposeIntoMatmul : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  static bool isTwoDTranspose(Value value) {
    auto transpose = value.getDefiningOp<linalg::TransposeOp>();
    if (!transpose)
      return false;
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    return permutation.size() == 2 && permutation[0] == 1 && permutation[1] == 0;
  }

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    bool changed = false;
    Value a = op.getA();
    Value b = op.getB();
    bool transposeA = static_cast<bool>(op.getTransposeA());
    bool transposeB = static_cast<bool>(op.getTransposeB());

    if (foldOperand(a, transposeA))
      changed = true;
    if (foldOperand(b, transposeB))
      changed = true;
    if (!changed)
      return failure();

    rewriter.modifyOpInPlace(op, [&]() {
      op.getAMutable().assign(a);
      op.getBMutable().assign(b);
      if (transposeA)
        op->setAttr("transpose_a", UnitAttr::get(rewriter.getContext()));
      else
        op->removeAttr("transpose_a");
      if (transposeB)
        op->setAttr("transpose_b", UnitAttr::get(rewriter.getContext()));
      else
        op->removeAttr("transpose_b");
    });
    return success();
  }

private:
  static bool foldOperand(Value &operand, bool &flag) {
    if (flag || !isTwoDTranspose(operand))
      return false;
    operand = operand.getDefiningOp<linalg::TransposeOp>().getInput();
    flag = true;
    return true;
  }
};

/// A matmul with no output elements becomes an empty tensor.
struct FoldZeroDimMatmul : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    auto outTy = op.getOutput().getType();
    if (outTy.getNumElements() != 0)
      return failure();
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, outTy.getShape(), outTy.getElementType());
    return success();
  }
};

/// Drop an explicit `scale = 1.0` from attention; 1.0 is the canonical
/// default and is implied when the attribute is absent.
struct CanonicalizeAttentionScale : public OpRewritePattern<AttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AttentionOp op, PatternRewriter &rewriter) const override {
    auto scale = op.getScaleAttr();
    if (!scale || scale.getValueAsDouble() != 1.0)
      return failure();
    rewriter.modifyOpInPlace(op, [&]() { op->removeAttr("scale"); });
    return success();
  }
};

/// Remove a schedule_region whose schedule is the identity by inlining its
/// body into the parent block.
struct InlineIdentityScheduleRegion : public OpRewritePattern<ScheduleRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleRegionOp op, PatternRewriter &rewriter) const override {
    for (Value result : op.getResults()) {
      auto ty = dyn_cast<RankedTensorType>(result.getType());
      if (!ty || !ty.hasStaticShape() || !op.getSchedule().isIdentity(ty.getShape()))
        return failure();
    }

    Block &body = op.getBody().front();
    auto yield = cast<YieldOp>(body.getTerminator());
    SmallVector<Value> yielded(yield.getValues());
    rewriter.inlineBlockBefore(&body, op);
    rewriter.replaceOp(op, yielded);
    rewriter.eraseOp(yield);
    return success();
  }
};

/// Collapse a nested schedule_region by composing the two schedules and
/// inlining the inner body.
struct CollapseNestedScheduleRegions : public OpRewritePattern<ScheduleRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ScheduleRegionOp outer, PatternRewriter &rewriter) const override {
    Block &outerBody = outer.getBody().front();
    ScheduleRegionOp inner;
    for (Operation &candidate : outerBody) {
      inner = dyn_cast<ScheduleRegionOp>(candidate);
      if (inner)
        break;
    }
    if (!inner)
      return failure();

    ScheduleAttr merged = ScheduleAttr::merge(outer.getSchedule(), inner.getSchedule());
    if (!merged)
      return failure();

    Block &innerBody = inner.getBody().front();
    auto innerYield = cast<YieldOp>(innerBody.getTerminator());
    SmallVector<Value> yielded(innerYield.getValues());
    rewriter.inlineBlockBefore(&innerBody, inner);
    rewriter.replaceOp(inner, yielded);
    rewriter.eraseOp(innerYield);
    rewriter.modifyOpInPlace(outer, [&]() { outer.setScheduleAttr(merged); });
    return success();
  }
};

} // namespace

void MatmulOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldTransposeIntoMatmul, FoldZeroDimMatmul>(context);
}

void AttentionOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<CanonicalizeAttentionScale>(context);
}

void ScheduleRegionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<InlineIdentityScheduleRegion, CollapseNestedScheduleRegions>(context);
}
