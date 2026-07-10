#include "seki/SekiAttrs.h"
#include "seki/SekiOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace {
// Returns true if op is a 3-loop (MxKxN) matrix contraction.
// Handles both named ops (linalg.matmul) and anonymous linalg.generic ops:
//    - Named ops implement ContractionOpInterface directly via ODS
//    - Generic ops requires a 2 step check:
//        1. isaContractionOpInterface: verifies the affine indexing maps form a contraction.
//        2. isaContractionBody: verifies the body contains a (mulf, addf) pair, which distinguishes
//           matmul from other 3-loop generics.
static bool isContraction2D(mlir::Operation *op) {
    // getNumLoops() counts total loops (parallel + reduction).
    // A 2D matmul has exactly 3: M (parallel), N (parallel), K (reduction).
    auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(op);
    if (!linalgOp || linalgOp.getNumLoops() != 3) return false;

    // Fast path, named ops (MatmulOp, MatmulTransposeBOp, etc.) carry the interface in their ODS definition.
    if (mlir::isa<mlir::linalg::ContractionOpInterface>(op)) return true;

    // Slow path for linalg.generics: check indexing maps then body ops.
    auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
    return generic && mlir::linalg::isaContractionOpInterface(generic) &&
        mlir::linalg::detail::isContractionBody(
            generic.getRegion().front(),
            [](mlir::Operation *elementwiseOp, mlir::Operation *reductionOp) {
                return mlir::isa<mlir::arith::MulFOp>(elementwiseOp) && mlir::isa<mlir::arith::AddFOp>(reductionOp);
            }
    );
}

// Same logic for a 4-loop (batchxMxKxN) contraction.
// The only difference is the loop count guard, 4 instead of 3.
static bool isContraction3D(mlir::Operation *op) {
    // getNumLoops() counts total loops (parallel + reduction).
    // A 3D matmul has exactly 4: batch(parallel), M (parallel), N (parallel), K (reduction).
    auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(op);
    if (!linalgOp || linalgOp.getNumLoops() != 4) return false;

    // Fast path, named ops (MatmulOp, MatmulTransposeBOp, etc.) carry the interface in their ODS definition.
    if (mlir::isa<mlir::linalg::ContractionOpInterface>(op)) return true;

    // Slow path for linalg.generics: check indexing maps then body ops.
    auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
    return generic && mlir::linalg::isaContractionOpInterface(generic) &&
        mlir::linalg::detail::isContractionBody(
            generic.getRegion().front(),
            [](mlir::Operation *elementwiseOp, mlir::Operation *reductionOp) {
                return mlir::isa<mlir::arith::MulFOp>(elementwiseOp) && mlir::isa<mlir::arith::AddFOp>(reductionOp);
            }
    );
}

// Detect the relu pattern: maximumf(input, 0.0) + linalg.yield.
// applyPatternsGreedily hoists arith.constant out of the block before our pattern runs,
// so the zero may be defined either inside (3-op body) or outside (2-op body).
// We scan by role rather than position and trace the zero via getDefiningOp().
static bool isReluBody(mlir::Block &body) {
    if (body.getNumArguments() != 2) return false;

    mlir::Operation *maxOp = nullptr;
    mlir::Operation *yieldOp = nullptr;
    for (mlir::Operation &op : body) {
        if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MaxNumFOp>(&op))
            maxOp = &op;
        else if (mlir::isa<mlir::linalg::YieldOp>(&op))
            yieldOp = &op;
        else if (!mlir::isa<mlir::arith::ConstantOp>(&op))
            return false; // unexpected op — not relu
    }
    if (!maxOp || !yieldOp) return false;

    // One operand must be block arg[0] (the input element); the other must be a 0.0 constant
    mlir::Value blockArg = body.getArgument(0);
    mlir::Value op0 = maxOp->getOperand(0);
    mlir::Value op1 = maxOp->getOperand(1);
    mlir::Value zeroCandidate;
    if (op0 == blockArg)      zeroCandidate = op1;
    else if (op1 == blockArg) zeroCandidate = op0;
    else return false;

    // zeroCandidate may be defined inside or outside the block — trace through getDefiningOp
    // getDefiningOp() can extract the constOp irrespective of whether it lives within the ^^bb0 block.
    // Constant folding might fold it to outside of the block
    auto constF = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(zeroCandidate.getDefiningOp());
    if (!constF) return false;
    auto fAttr = mlir::dyn_cast<mlir::FloatAttr>(constF.getValue());
    if (!fAttr || !fAttr.getValue().isZero()) return false;

    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(yieldOp);
    return yield && yield.getNumOperands() == 1 &&
           yield.getOperand(0) == maxOp->getResult(0);
}

// Returns the UnaryFn if the body is a single unary elementwise op, else nullopt.
static std::optional<mlir::seki::UnaryFn> matchUnaryBody(mlir::Block &body) {
    // UnaryFn always have 2 arguments, Seki_UnaryFnAttr:$fn and AnyTensor:$input
    // If it doesn't have 2 arguments, simply return std::nullopt as it doesn't match this pattern 
    if (body.getNumArguments() != 2) return std::nullopt;

    // Check relu (3-op special case) first, then the simple 2-op single-input ops
    if (isReluBody(body)) return mlir::seki::UnaryFn::relu;

    // Remaining unary ops have exactly 2 ops: computeOp + yield
    // E.g. linalg.generic ... {
    // ^bb0(%in: f32, %out: f32):
    //     %result = math.exp %in : f32
    //     linalg.yield %result : f32
    // }
    auto it = body.begin();
    mlir::Operation *computeOp = &*it++; // points at : %result = math.exp %in
    mlir::Operation *yieldOp   = &*it++; // points at : linalg.yield %result
    if (it != body.end()) return std::nullopt; // If it has more than the targeted 2 ops, return as it doesn't fit pattern

    // computeOp like math.exp %in has exactly one operand
    if (computeOp->getNumOperands() != 1) return std::nullopt;
    // arg[0] is always the input element in the body e.g. ^bb0(%in: f32, %out: f32)
    // Check that the same SSA is consumed in the computeOp
    if (computeOp->getOperand(0) != body.getArgument(0)) return std::nullopt;

    // yield must forward computeOp's result
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(yieldOp);
    // Check that yield exists and then it has exactly 1 operand which should be the result returned by the computeOp
    if (!yield || yield.getNumOperands() != 1) return std::nullopt;
    if (yield.getOperand(0) != computeOp->getResult(0)) return std::nullopt;

    // Map math.(computeOp) to seki's UnaryFn attr
    if (mlir::isa<mlir::math::ExpOp>(computeOp))    return mlir::seki::UnaryFn::exp;
    if (mlir::isa<mlir::math::LogOp>(computeOp))    return mlir::seki::UnaryFn::log;
    if (mlir::isa<mlir::arith::NegFOp>(computeOp))    return mlir::seki::UnaryFn::neg;
    if (mlir::isa<mlir::math::AbsFOp>(computeOp))    return mlir::seki::UnaryFn::abs;

    return std::nullopt;
}

// Returns the BinaryFn if the body is a simple binary elementwise op, else nullopt.
// BinaryFn has 3 blocks args: Seki_BinaryFnAttr:$fn, AnyTensor:$lhs and AnyTensor:$rhs
static std::optional<mlir::seki::BinaryFn> matchBinaryBody(mlir::Block &body) {
    if (body.getNumArguments() != 3)  return std::nullopt;
    // E.g linalg.generic ins(%lhs, %rhs : tensor<Nxf32>, tensor<Nxf32>)
    //                    outs(%out : tensor<Nxf32>) {
    //    ^bb0(%l: f32, %r: f32, %o: f32)
    //        %result = arith.addf %l, %r
    //        linalg.yield %result
    //     }
    auto it = body.begin();
    mlir::Operation *computeOp = &*it++;
    mlir::Operation *yieldOp = &*it++;
    if (it != body.end()) return std::nullopt;

    // computeOp consumes lhs(arg[0]) then rhs (arg[1]) in that order.
    // Linalg lowering always place the lhs first, no need to check swap.
    if (computeOp->getNumOperands() != 2) return std::nullopt;
    if (computeOp->getOperand(0) != body.getArgument(0)) return std::nullopt;
    if (computeOp->getOperand(1) != body.getArgument(1)) return std::nullopt;

    // Check that yield exists, and it has operand coming from the result of the computeOp
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(yieldOp);
    if (!yield || yield.getNumOperands() != 1) return std::nullopt;
    if (yield.getOperand(0) != computeOp->getResult(0)) return std::nullopt;

    // Map math.(computeOp) to seki's BinaryFn attr
    if (mlir::isa<mlir::arith::AddFOp>(computeOp)) return mlir::seki::BinaryFn::add;
    if (mlir::isa<mlir::arith::SubFOp>(computeOp)) return mlir::seki::BinaryFn::sub;
    if (mlir::isa<mlir::arith::MulFOp>(computeOp)) return mlir::seki::BinaryFn::mul;
    if (mlir::isa<mlir::arith::DivFOp>(computeOp)) return mlir::seki::BinaryFn::div;
    if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MaxNumFOp>(computeOp)) return mlir::seki::BinaryFn::max;
    if (mlir::isa<mlir::arith::MinimumFOp, mlir::arith::MinNumFOp>(computeOp)) return mlir::seki::BinaryFn::min;

    return std::nullopt;
}

// Returns the ReduceFn if the body is a reduction, else nullopt
static std::optional<mlir::seki::ReduceFn> matchReduceBody(mlir::Block &body) {
    if (body.getNumArguments() != 2) return std::nullopt; // 2 block args, input elem and acc (running sum for example)
    // E.g. linalg.generic ins(%input : tensor<128x128xf32)
    //                     outs(%init : tensor<128xf32>) {
    //      ^bb0(%in: f32, %acc: f32):
    //          %result = arith.addf %in, %acc // %acc correspond to one element from %init (the output buffer / running accumulator)
    //           linalg.yield %result
    //      }
    auto it = body.begin();
    mlir::Operation *computeOp = &*it++;
    mlir::Operation *yieldOp = &*it++;
    if (it != body.end()) return std::nullopt;

    // ComputeOp should have 2 op, operating on one element from %init, and the input operand
    if (computeOp->getNumOperands() != 2) return std::nullopt;
    mlir::Value in = body.getArgument(0);
    mlir::Value acc = body.getArgument(1);
    // Check the body arguments are consumed in the computeOp
    // %in and %acc of ^bb0 is consumed in arith.addf
    bool usesInAndAcc = (computeOp->getOperand(0) == in && computeOp->getOperand(1) == acc) ||
                        (computeOp->getOperand(0) == acc && computeOp->getOperand(1) == in);
    if (!usesInAndAcc) return std::nullopt;

    // Check that yield exists, and it has operand coming from the result of the computeOp
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(yieldOp);
    if (!yield || yield.getNumOperands() != 1) return std::nullopt;
    if (yield.getOperand(0) != computeOp->getResult(0)) return std::nullopt;

    // Match computeOp to ReduceFn attr
    if (mlir::isa<mlir::arith::AddFOp>(computeOp)) return mlir::seki::ReduceFn::sum;
    if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MaxNumFOp>(computeOp)) return mlir::seki::ReduceFn::max;
    if (mlir::isa<mlir::arith::MinimumFOp, mlir::arith::MinNumFOp>(computeOp)) return mlir::seki::ReduceFn::min;

    return std::nullopt;
}

// Lifts named ops of linalg, e.g. matmul, matmul_tranposed_b, batch_matmul to the matching seki op through ContractionOpInterface
struct NamedContractionLiftingPattern : mlir::OpInterfaceRewritePattern<mlir::linalg::ContractionOpInterface> {
    using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::linalg::ContractionOpInterface op, mlir::PatternRewriter &rewriter) const override {
        // linalg.generic uses linalg::isaContractionOpInterface() for runtime checks but does not implement ContractionOpInterface,
        // it is handled by GenericLiftingPattern
        // op.getOperation() unwraps the interface to Operation* for isa/cast
        if (mlir::isa<mlir::linalg::GenericOp>(op.getOperation())) return mlir::failure();

        auto linalgOp = mlir::cast<mlir::linalg::LinalgOp>(op.getOperation());
        mlir::Value A = linalgOp.getDpsInputs()[0];
        mlir::Value B = linalgOp.getDpsInputs()[1];
        mlir::Value C = linalgOp.getDpsInits()[0];

        // transpose_b : set UnitAttr if the B operand is transposed
        // op.getOperation() unwraps the interface to the raw Operation* that mlir::isa<> requires
        auto transposeB = mlir::isa<mlir::linalg::MatmulTransposeBOp, mlir::linalg::BatchMatmulTransposeBOp>(op.getOperation()) ?
                          mlir::UnitAttr::get(rewriter.getContext()) : mlir::UnitAttr{};

        // Check the iteration space. Each linalg op describes a set of nested loops over index variables. getNumLoops() returns how many
        // loop dimensions exist.
        if (linalgOp.getNumLoops() == 3) {
            rewriter.replaceOpWithNewOp<mlir::seki::MatmulOp>(
                linalgOp, C.getType(), A, B, C, transposeB);
            return mlir::success();
        }
        if (linalgOp.getNumLoops() == 4) {
            rewriter.replaceOpWithNewOp<mlir::seki::BatchMatmulOp>(
                linalgOp, C.getType(), A, B, C, transposeB);
            return mlir::success();
        }
        return mlir::failure();
    }
};

// Lifts linalg.generic to the matching seki op by dispatching on body structure.
// Unrecognised bodies emit a compile-time error at the op's source location.
struct GenericLiftingPattern : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::Block &body  = op.getRegion().front();

        // --- 2D matmul: 3-loop (MxKxN) contraction -> seki.matmul ---
        if (isContraction2D(op)) {
            // getDpsInputs() = ins(...) tensors; getDpsInits() = outs(...) tensors.
            // For matmul: inputs[0] = A, inputs[1] = B, inits[0] = C
            mlir::Value A = op.getDpsInputs()[0];
            mlir::Value B = op.getDpsInputs()[1];
            mlir::Value C = op.getDpsInits()[0];
            // UnitAttr{} = null = transpose_b absent.
            rewriter.replaceOpWithNewOp<mlir::seki::MatmulOp>(
                op, C.getType(), A, B, C, mlir::UnitAttr{}
            );
            return mlir::success();
        }
        // --- 3D matmul: 4-loop (batchxMxKxN) contraction -> seki.batch_matmul ---
        if (isContraction3D(op)) {
            mlir::Value A = op.getDpsInputs()[0];
            mlir::Value B = op.getDpsInputs()[1];
            mlir::Value C = op.getDpsInits()[0];
            rewriter.replaceOpWithNewOp<mlir::seki::BatchMatmulOp>(
                op, C.getType(), A, B, C, mlir::UnitAttr{}
            );
            return mlir::success();
        }

        // --- Elementwise unary -> seki.ew_unary<fn> ---
        if (auto fn = matchUnaryBody(body)) {
            mlir::Value input = op.getDpsInputs()[0];
            mlir::Value output = op.getDpsInits()[0];
            // Get the associated FnAttr in fn which is returned by matchUnaryBody
            auto fnAttr = mlir::seki::UnaryFnAttr::get(op.getContext(), *fn);
            rewriter.replaceOpWithNewOp<mlir::seki::EwUnaryOp>(
                op, output.getType(), fnAttr, input
            );
            return mlir::success();
        }

        // --- Elementwise binary -> seki.ew_binary<fn> ---
        if (auto fn = matchBinaryBody(body)) {
            mlir::Value lhs = op.getDpsInputs()[0];
            mlir::Value rhs = op.getDpsInputs()[1];
            mlir::Value output = op.getDpsInits()[0];
            // Get the associated FnAttr in fn which is returned by matchBinaryBody
            auto fnAttr = mlir::seki::BinaryFnAttr::get(op.getContext(), *fn);
            rewriter.replaceOpWithNewOp<mlir::seki::EwBinaryOp>(
                op, output.getType(), fnAttr, lhs, rhs
            );
            return mlir::success();
        }

        // --- Reduction -> seki.reduce<fn> axis = N ---
        if (auto fn = matchReduceBody(body)) {
            // Scan iterator types to find the reduction loop index -> that is the axis.
            // E.g. iterator_types = ["parallel", "reduction"] -> axis = 1.
            int64_t axis = -1;
            auto iterTypes = op.getIteratorTypesArray(); // Return ["parallel", "reduction"]
            // Loop through the array, find the first reduction iterator
            for (int64_t i = 0; i < static_cast<int64_t>(iterTypes.size()); ++i) {
                if (iterTypes[i] == mlir::utils::IteratorType::reduction) {
                    axis = i; // Set it as the reduction axis
                    break;
                }
            }

            // if there are no "reduction" in types array, return failure
            if (axis == -1) return mlir::failure();

            mlir::Value input = op.getDpsInputs()[0];
            mlir::Value output = op.getDpsInits()[0];
            // Get the ReduceFnAttr returned by matchReduceBody
            auto fnAttr = mlir::seki::ReduceFnAttr::get(op.getContext(), *fn);
            // getI64IntegerAttr wraps the int64_t as an MLIR I64Attr for the axis parameter.
            rewriter.replaceOpWithNewOp<mlir::seki::ReduceOp>(
                op, output.getType(), fnAttr, input, rewriter.getI64IntegerAttr(axis)
            );
            return mlir::success();
        }

        // --- Unrecognised: pin error to source location, pass runner signals failure -- 
        op.emitError("Unrecognised linalg.generic body; cannot lift to seki dialect");
        return mlir::failure();
    }
};

// linalg.broadcast { dimensions = [...] -> seki.broadcast broadcast_dims = [...]}
struct BroadcastLiftingPattern : mlir::OpRewritePattern<mlir::linalg::BroadcastOp> {
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::linalg::BroadcastOp op, mlir::PatternRewriter &rewriter) const override {
        // getDenseI64ArrayAttr wraps ArrayRef<int64_t> into the DenseI64ArrayAttr that seki.broadcast's broadcast_dims 
        // parameter expects.
        rewriter.replaceOpWithNewOp<mlir::seki::BroadcastOp>(
            op, op.getInit().getType(), op.getInput(), rewriter.getDenseI64ArrayAttr(op.getDimensions())
        );
        return mlir::success();
    }
};

// linalg.tranpose { permutation = [...]} -> seki.tranpose permutation = [...]
struct TransposeLiftingPattern : mlir::OpRewritePattern<mlir::linalg::TransposeOp> {
    using OpRewritePattern::OpRewritePattern;

    // seki.reshape doesn't have the reassociation map, the input type and result type together fully specify the transformation
    mlir::LogicalResult matchAndRewrite(mlir::linalg::TransposeOp op, mlir::PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<mlir::seki::TransposeOp>(op, op.getInit().getType(), op.getInput(), 
                                                            rewriter.getDenseI64ArrayAttr(op.getPermutation()));
        return mlir::success();
    }
};

// tensor.collapse_shape -> seki.reshape
// E.g. %collapsed = tensor.collapse_shape %input [[0], [1, 2]] : tensor<4x8x16xf32> into tensor<4x128xf32>
// [[0], [1,2]] means output dim0 <- input dim 0 alone. Output dim 1 <- input dims 1 and 2 merged
struct CollapseShapeLiftingPattern : mlir::OpRewritePattern<mlir::tensor::CollapseShapeOp> {
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::tensor::CollapseShapeOp op, mlir::PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<mlir::seki::ReshapeOp>(op, op.getResultType(), op.getSrc());
        return mlir::success();
    }
};

// tensor.expand_shape -> seki.reshape
// E.g. %expanded = tensor.expand_shape %input [[0], [1,2]]
//                     output_shape [4, 8, 16] : tensor<4x128xf32> into tensor<4x8x16xf32>
// Expand a 2D tensor into 3D by splitting the last dim
// [batch, rows*col] -> [batch, rows, cols]
struct ExpandShapeLiftingPattern : mlir::OpRewritePattern<mlir::tensor::ExpandShapeOp> {
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::tensor::ExpandShapeOp op, mlir::PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<mlir::seki::ReshapeOp>(op, op.getResultType(), op.getSrc());
        return mlir::success();
    }
};

} // namespace

void populateSekiLiftingPatterns(mlir::RewritePatternSet &patterns) {
    patterns.add<
        NamedContractionLiftingPattern,
        GenericLiftingPattern,
        BroadcastLiftingPattern,
        TransposeLiftingPattern,
        CollapseShapeLiftingPattern,
        ExpandShapeLiftingPattern
    >(patterns.getContext());
}
