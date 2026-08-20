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

        // Build the attribute list by copying all existing matmul attributes
        // and add transpose_b = unit.
        mlir::NamedAttrList attrs(op->getAttrs());
        attrs.set(rewriter.getStringAttr("transpose_b"), rewriter.getUnitAttr());

        // Replace with a new matmul whose B operand is the transpose's input.
        // seki.tranpose gets eliminated 
        rewriter.replaceOpWithNewOp<MatmulOp>(
            op,
            op.getResult().getType(), // result type unchanged
            op.getA(),                // A operand unchanged
            transposeOp.getInput(),   // B operand: bypass the transpose
            op.getC(),                // C operand unchanged
            attrs.getDictionary(rewriter.getContext())
        );
        return success();
    }
};

void populateTransposeMatmulFusionPatterns(RewritePatternSet &patterns) {
    patterns.add<TransposeMatmulFusionPattern>(patterns.getContext());
}
} // namespace mlir::seki
