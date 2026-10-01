#include "seki/SekiFusionPatterns.h"
#include "seki/SekiOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::seki {
// Folds:
//   %K_T = seki.transpose %K permutation = [1, 0] : tensor<MxKxT> -> tensor<KxMxT>
//   %C = seki.matmul ins(%Q, %K_T : ...) -> ...
// Into:
//   %C = seki.matmul ins(%Q, %K : ...) outs(%acc : ...) {transpose_b} -> ...
struct TransposeMatmulFusionPattern : OpRewritePattern<MatmulOp> {
    using OpRewritePattern<MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
        // B operand must come from seki.transpose
        Value B = op.getB();
        auto transposeOp = B.getDefiningOp<TransposeOp>();
        if (!transposeOp)
            return failure();

        // We only fold transpose + matmul -> matmul {transposed_b} when transposed B is used
        // exactly once
        if (!B.hasOneUse())
            return failure();

        // Replace with a new matmul whose B operand is the transpose's input.
        // seki.tranpose gets eliminated 
        rewriter.replaceOpWithNewOp<MatmulOp>(
            op,
            op.getResult().getType(), // result type unchanged
            op.getA(),                // A operand unchanged
            transposeOp.getInput(),   // B operand: bypass the transpose
            op.getC(),                // C operand unchanged
            rewriter.getUnitAttr()     // transpose_b
        );
        return success();
    }
};

void populateTransposeMatmulFusionPatterns(RewritePatternSet &patterns) {
    patterns.add<TransposeMatmulFusionPattern>(patterns.getContext());
}

// Fuses a single-op epilogue into a seki.fused_computation region:
//   %t = seki.matmul ins(%A, %B) outs(%C) -> T
//   %r = seki.ew_unary<fn> %t : T
// becomes:
//   %r = seki.fused_computation(%A, %B, %C : ...) -> T {
//   ^bb0(%a, %b, %c):
//     %t = seki.matmul ins(%a, %b) outs(%c) -> T
//     %rr = seki.ew_unary<fn> %t : T
//     seki.yield %tt : T
//   }
struct SingleOpEpilogueFusionPattern : OpRewritePattern<EwUnaryOp> {
    using OpRewritePattern<EwUnaryOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(EwUnaryOp op, PatternRewriter &rewriter) const override {
        // Never fuse inside an existing fused_computation region, groups are flat by design.
        // Without this, applyPatternsGreedily revists the cloned ops inside the newly created region and
        // fires the pattern again, nesting the fused_computation ops infinitely.
        if (op->getParentOfType<FusedComputationOp>())
            return failure();

        // Input must come from seki.matmul
        auto matmulOp = op.getInput().getDefiningOp<MatmulOp>();
        if (!matmulOp)
            return failure();

        // Closed group condition guard: matmul result has exactly one use (this ew_unary).
        // If it escaped to a second consumer, fusing would silently drop that consumer's input.
        if (!matmulOp.getResult().hasOneUse())
            return failure();

        // The fusion group's external inputs are the mamul's three operands.
        llvm::SmallVector<Value> inputs = {matmulOp.getA(), matmulOp.getB(), matmulOp.getC()};
        llvm::SmallVector<Type> inputTypes;
        for (Value v: inputs)
            inputTypes.push_back(v.getType());

        // Create the region op. Its result type is the ew_unary's result type
        auto fused = FusedComputationOp::create(
            rewriter, op.getLoc(), op.getResult().getType(), inputs
        );

        // Build the single block with block arguments %a, %b, %c with the same types as %A, %B, %C.
        Block *block = rewriter.createBlock(&fused.getBody());
        for (Type t : inputTypes)
            block->addArgument(t, op.getLoc());

        // IRMapping: outer values (%A, %B, %C) -> block arguments (%a, %b, %c).
        // Cloning the matmul with this mapping rewires its operands automatically
        // %A -> %a, %B -> %b, %C -> %c
        IRMapping mapping;
        for (auto [outer, arg] : llvm::zip(inputs, block->getArguments()))
            mapping.map(outer, arg);

        rewriter.setInsertionPointToStart(block);
        // The original matmul's operands are %A, %B, %c. clone() looks each on up in mapping, finds all three,
        // and subsitutes them.  The clone becomes:
        // %t' = seki.matmul ins(%a, %b: T1, T2) outs(%c: T3) -> T3
        Operation *clonedMatmul = rewriter.clone(*matmulOp, mapping);
        // Map the matmul's old result to the clone's result, so cloning the ew_unary next picks up
        // the in-region matmul output.
        // This adds the rule %t -> %t'. Without this, the next clone step would fail to find %t in the mapping and
        // keep pointing at the original outer %t, which lives outside the region.
        mapping.map(matmulOp.getResult(), clonedMatmul->getResult(0));
        // clone the current ew_unary operand. %r' = seki.ew_unary<relu> %t' : T3
        Operation *clonedEwUnary = rewriter.clone(*op, mapping);

        // Terminates the block with seki.yield %r' : T3
        YieldOp::create(rewriter, op.getLoc(), clonedEwUnary->getResult(0));

        // End state inside the region
        // ^bb0(%a: T1, %b: T2, %c: T3):
        //   %t' = seki.matmul ins(%a, %b : T1, T2) outs(%c : T3) -> T3
        //   %r' = seki.ew_unary<relu> %t' : T3
        //   seki.yield %r' : T3

        // Replace the original ew_unary with the fused region's result.
        rewriter.replaceOp(op, fused.getResult());

        // MatmulOp has no Pure trait, so applyPatternsGreedily's DCE won't remove it
        // automatically even with zero use, erase the matmulOp explicitly.
        rewriter.eraseOp(matmulOp);
        return success();
    }
};

void populateSingleOpEpilogueFusionPatterns(RewritePatternSet &patterns) {
    patterns.add<SingleOpEpilogueFusionPattern>(patterns.getContext());
}

} // namespace mlir::seki
