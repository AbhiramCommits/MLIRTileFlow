#include "TileFlow/Dialect/TileFlowOps.h"
#include "TileFlow/Transforms/Passes.h"
#include "TileFlow/Transforms/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"


namespace mlir {
namespace tileflow {
#define GEN_PASS_DEF_TILEFLOWTOLINALG
#include "TileFlow/Transforms/Passes.h.inc"
} // namespace tileflow
} // namespace mlir

using namespace mlir;
using namespace mlir::tileflow;

namespace {

static Value createEmpty(OpBuilder &builder, Location loc, RankedTensorType type) {
  return tensor::EmptyOp::create(builder, loc, type.getShape(), type.getElementType());
}

static Value createScalar(OpBuilder &builder, Location loc, Type elementType, double value) {
  return arith::ConstantOp::create(builder, loc, builder.getFloatAttr(elementType, value));
}

/// A large finite negative value usable as "-infinity" without emitting the
/// `float -inf` literal that some LLVM IR parsers reject.
static double negativeSentinel(Type elementType) {
  return elementType.isF16() ? -60000.0 : -1.0e30;
}

static Value createFilled(OpBuilder &builder, Location loc, RankedTensorType type, Value scalar) {
  Value empty = createEmpty(builder, loc, type);
  return linalg::FillOp::create(builder, loc, TypeRange{type}, ValueRange{scalar},
                                ValueRange{empty})
      .getResults()
      .front();
}

/// Derive a schedule over a subset/permutation of the source tile sizes.
/// The derived schedule always uses the identity loop order.
static ScheduleAttr deriveSchedule(ScheduleAttr source, ArrayRef<int64_t> tileIndices) {
  if (!source)
    return nullptr;
  ArrayRef<int64_t> tiles = source.getTileSizes();
  SmallVector<int64_t> selected;
  for (int64_t index : tileIndices) {
    if (index < 0 || index >= static_cast<int64_t>(tiles.size()))
      return nullptr;
    selected.push_back(tiles[index]);
  }
  SmallVector<int64_t> order;
  for (int64_t i = 0; i < static_cast<int64_t>(selected.size()); ++i)
    order.push_back(i);
  return ScheduleAttr::get(source.getContext(), selected, order, source.getUnroll(),
                           source.getPromoteNames(), source.getTarget().getValue());
}

/// Build softmax along the last dimension of a rank-N tensor.
static Value buildSoftmaxLastAxis(OpBuilder &builder, Location loc, Value input,
                                  ScheduleAttr schedule) {
  auto inputTy = cast<RankedTensorType>(input.getType());
  int64_t rank = inputTy.getRank();
  MLIRContext *ctx = builder.getContext();
  Type elementType = inputTy.getElementType();

  SmallVector<int64_t> reduceShape(inputTy.getShape().drop_back());
  auto reduceTy = RankedTensorType::get(reduceShape, elementType);

  AffineMap identity = AffineMap::getMultiDimIdentityMap(rank, ctx);
  SmallVector<AffineExpr> reduceExprs;
  for (int64_t i = 0; i < rank - 1; ++i)
    reduceExprs.push_back(builder.getAffineDimExpr(i));
  AffineMap reduceMap = AffineMap::get(rank, 0, reduceExprs, ctx);

  SmallVector<utils::IteratorType> reduceIters(rank, utils::IteratorType::parallel);
  reduceIters.back() = utils::IteratorType::reduction;

  // max(x)
  Value maxInit =
      createFilled(builder, loc, reduceTy,
                   createScalar(builder, loc, elementType, negativeSentinel(elementType)));
  auto maxOp = linalg::GenericOp::create(
      builder, loc, TypeRange{reduceTy}, ValueRange{input}, ValueRange{maxInit},
      ArrayRef<AffineMap>{identity, reduceMap}, reduceIters,
      [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value max = arith::MaximumFOp::create(nested, nestedLoc, args[0], args[1]);
        linalg::YieldOp::create(nested, nestedLoc, max);
      });
  attachSchedule(maxOp, schedule);

  // exp(x - max)
  Value expInit = createEmpty(builder, loc, inputTy);
  auto expOp = linalg::GenericOp::create(
      builder, loc, TypeRange{inputTy}, ValueRange{input, maxOp.getResults().front()},
      ValueRange{expInit}, ArrayRef<AffineMap>{identity, reduceMap, identity},
      SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
      [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value shifted = arith::SubFOp::create(nested, nestedLoc, args[0], args[1]);
        Value exp = math::ExpOp::create(nested, nestedLoc, shifted);
        linalg::YieldOp::create(nested, nestedLoc, exp);
      });
  attachSchedule(expOp, schedule);

  // sum(exp)
  Value sumInit =
      createFilled(builder, loc, reduceTy, createScalar(builder, loc, elementType, 0.0));
  auto sumOp = linalg::GenericOp::create(
      builder, loc, TypeRange{reduceTy}, ValueRange{expOp.getResults().front()},
      ValueRange{sumInit}, ArrayRef<AffineMap>{identity, reduceMap}, reduceIters,
      [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value sum = arith::AddFOp::create(nested, nestedLoc, args[0], args[1]);
        linalg::YieldOp::create(nested, nestedLoc, sum);
      });
  attachSchedule(sumOp, schedule);

  // exp / sum
  Value divInit = createEmpty(builder, loc, inputTy);
  auto divOp = linalg::GenericOp::create(
      builder, loc, TypeRange{inputTy},
      ValueRange{expOp.getResults().front(), sumOp.getResults().front()}, ValueRange{divInit},
      ArrayRef<AffineMap>{identity, reduceMap, identity},
      SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
      [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value quotient = arith::DivFOp::create(nested, nestedLoc, args[0], args[1]);
        linalg::YieldOp::create(nested, nestedLoc, quotient);
      });
  attachSchedule(divOp, schedule);

  return divOp.getResults().front();
}

static Value buildTranspose(OpBuilder &builder, Location loc, Value input,
                            ArrayRef<int64_t> permutation) {
  auto inputTy = cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> shape;
  for (int64_t index : permutation)
    shape.push_back(inputTy.getDimSize(index));
  auto resultTy = RankedTensorType::get(shape, inputTy.getElementType());
  Value empty = createEmpty(builder, loc, resultTy);
  return linalg::TransposeOp::create(builder, loc, input, empty, permutation).getResults().front();
}

static Value buildMatmul(OpBuilder &builder, Location loc, Value a, Value b,
                         RankedTensorType resultTy) {
  // A matmul accumulates into its destination, so the destination must be
  // zero-initialized rather than undefined (tensor.empty).
  Value zero = createScalar(builder, loc, resultTy.getElementType(), 0.0);
  Value init = createFilled(builder, loc, resultTy, zero);
  return linalg::MatmulOp::create(builder, loc, TypeRange{resultTy}, ValueRange{a, b},
                                  ValueRange{init})
      .getResults()
      .front();
}

/// Lower tileflow.matmul to linalg.transpose (when needed) + linalg.matmul.
static Value lowerMatmul(OpBuilder &builder, Location loc, MatmulOp op) {
  Value a = op.getA();
  Value b = op.getB();
  if (op.getTransposeA())
    a = buildTranspose(builder, loc, a, {1, 0});
  if (op.getTransposeB())
    b = buildTranspose(builder, loc, b, {1, 0});
  auto resultTy = cast<RankedTensorType>(op.getOutput().getType());
  Value result = buildMatmul(builder, loc, a, b, resultTy);
  attachSchedule(result.getDefiningOp(), op.getScheduleAttr());
  return result;
}

static Value lowerAttention(OpBuilder &builder, Location loc, AttentionOp op) {
  auto qTy = cast<RankedTensorType>(op.getQ().getType());
  auto kTy = cast<RankedTensorType>(op.getK().getType());
  auto vTy = cast<RankedTensorType>(op.getV().getType());
  Type elementType = qTy.getElementType();
  ScheduleAttr schedule = op.getScheduleAttr();

  int64_t heads = qTy.getDimSize(0);
  int64_t seqQ = qTy.getDimSize(1);
  int64_t seqK = kTy.getDimSize(1);
  int64_t valueDim = vTy.getDimSize(2);

  // K^T: [H, Sk, D] -> [H, D, Sk].
  Value kTransposed = buildTranspose(builder, loc, op.getK(), {0, 2, 1});

  // QK^T: [H, Sq, Sk]; linalg.batch_matmul iterates [batch, M, N, K].
  auto scoresTy = RankedTensorType::get({heads, seqQ, seqK}, elementType);
  Value scoresZero = createScalar(builder, loc, elementType, 0.0);
  Value scoresEmpty = createFilled(builder, loc, scoresTy, scoresZero);
  auto qk =
      linalg::BatchMatmulOp::create(builder, loc, TypeRange{scoresTy},
                                    ValueRange{op.getQ(), kTransposed}, ValueRange{scoresEmpty});
  attachSchedule(qk, deriveSchedule(schedule, {0, 1, 2, 3}));
  Value scores = qk.getResults().front();

  AffineMap identity3 = AffineMap::getMultiDimIdentityMap(3, builder.getContext());

  // Scale.
  double scale = op.getScaleValue();
  if (scale != 1.0) {
    Value scaleCst = createScalar(builder, loc, elementType, scale);
    Value scaledEmpty = createEmpty(builder, loc, scoresTy);
    auto scaleOp = linalg::GenericOp::create(
        builder, loc, TypeRange{scoresTy}, ValueRange{scores}, ValueRange{scaledEmpty},
        ArrayRef<AffineMap>{identity3, identity3},
        SmallVector<utils::IteratorType>(3, utils::IteratorType::parallel),
        [scaleCst](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value scaled = arith::MulFOp::create(nested, nestedLoc, args[0], scaleCst);
          linalg::YieldOp::create(nested, nestedLoc, scaled);
        });
    attachSchedule(scaleOp, deriveSchedule(schedule, {0, 1, 2}));
    scores = scaleOp.getResults().front();
  }

  // Causal mask: score[h, i, j] = -inf when j > i.
  if (op.getCausalFlag()) {
    Value negInf = createScalar(builder, loc, elementType, negativeSentinel(elementType));
    Value maskedEmpty = createEmpty(builder, loc, scoresTy);
    auto maskOp = linalg::GenericOp::create(
        builder, loc, TypeRange{scoresTy}, ValueRange{scores}, ValueRange{maskedEmpty},
        ArrayRef<AffineMap>{identity3, identity3},
        SmallVector<utils::IteratorType>(3, utils::IteratorType::parallel),
        [negInf](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value i = linalg::IndexOp::create(nested, nestedLoc, 1);
          Value j = linalg::IndexOp::create(nested, nestedLoc, 2);
          Value past = arith::CmpIOp::create(nested, nestedLoc, arith::CmpIPredicate::sgt, j, i);
          Value masked = arith::SelectOp::create(nested, nestedLoc, past, negInf, args[0]);
          linalg::YieldOp::create(nested, nestedLoc, masked);
        });
    attachSchedule(maskOp, deriveSchedule(schedule, {0, 1, 2}));
    scores = maskOp.getResults().front();
  }

  // Softmax over the key dimension.
  scores = buildSoftmaxLastAxis(builder, loc, scores, deriveSchedule(schedule, {0, 1, 2}));

  // PV: [H, Sq, Sk] x [H, Sk, Dv] -> [H, Sq, Dv]. The schedule order for a
  // batch matmul is [batch, M, N, K] = [H, Sq, Dv, Sk].
  auto resultTy = RankedTensorType::get({heads, seqQ, valueDim}, elementType);
  Value resultZero = createScalar(builder, loc, elementType, 0.0);
  Value resultEmpty = createFilled(builder, loc, resultTy, resultZero);
  auto pv = linalg::BatchMatmulOp::create(builder, loc, TypeRange{resultTy},
                                          ValueRange{scores, op.getV()}, ValueRange{resultEmpty});
  attachSchedule(pv, deriveSchedule(schedule, {0, 1, 3, 2}));
  return pv.getResults().front();
}

static Value lowerSoftmaxRow(OpBuilder &builder, Location loc, SoftmaxRowOp op) {
  ScheduleAttr schedule = op.getScheduleAttr();
  if (op.getAxisValue() == 0) {
    // Transpose so the reduction is over the last dimension.
    Value input = buildTranspose(builder, loc, op.getInput(), {1, 0});
    ScheduleAttr swapped;
    if (schedule) {
      ArrayRef<int64_t> tiles = schedule.getTileSizes();
      SmallVector<int64_t> swappedTiles{tiles[1], tiles[0]};
      SmallVector<int64_t> order{0, 1};
      swapped = ScheduleAttr::get(schedule.getContext(), swappedTiles, order, schedule.getUnroll(),
                                  schedule.getPromoteNames(), schedule.getTarget().getValue());
    }
    Value softmaxed = buildSoftmaxLastAxis(builder, loc, input, swapped);
    return buildTranspose(builder, loc, softmaxed, {1, 0});
  }
  return buildSoftmaxLastAxis(builder, loc, op.getInput(), schedule);
}

struct TileFlowToLinalgPass
    : public mlir::tileflow::impl::TileFlowToLinalgBase<TileFlowToLinalgPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<Operation *> worklist;
    module.walk([&](Operation *op) {
      if (isa<MatmulOp, SoftmaxRowOp, AttentionOp>(op))
        worklist.push_back(op);
    });

    for (Operation *op : worklist) {
      rewriter.setInsertionPoint(op);
      Location loc = op->getLoc();
      Value lowered;
      if (auto matmul = dyn_cast<MatmulOp>(op))
        lowered = lowerMatmul(rewriter, loc, matmul);
      else if (auto softmax = dyn_cast<SoftmaxRowOp>(op))
        lowered = lowerSoftmaxRow(rewriter, loc, softmax);
      else if (auto attention = dyn_cast<AttentionOp>(op))
        lowered = lowerAttention(rewriter, loc, attention);
      if (!lowered) {
        op->emitError("tileflow lowering failed");
        signalPassFailure();
        return;
      }
      rewriter.replaceOp(op, lowered);
    }
  }
};

} // namespace
