#include "seki/SekiTilingPatterns.h"
#include "seki/SekiOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::seki {
// Commutes extract_slice(cast(A)) -> cast(extract_slice(A)).
// This allows for loading casted directly from memory to avoid operating
// cast in vec engine.
// For example:
// %0 = seki.cast %arg0 : tensor<256x512xf32> -> tensor<256x512xbf16>
// scf.for %arg3 = ... {
//   scf.for %arg5 = ... {
//     %slice = tensor.extract_slice %0[%arg3, %arg5] [128, 128] [1, 1]
//              : tensor<256x512xbf16> to tensor<128x128xbf16>
//     ...
//   }
// }
// gets turned into:
// scf.for %arg3 = ... {
//   scf.for %arg5 = ... {
//     %preslice = tensor.extract_slice %arg0[%arg3, %arg5] [128, 128] [1, 1]
//                 : tensor<256x512xf32> to tensor<128x128xf32>
//     %slice = seki.cast %preslice : tensor<128x128xf32> -> tensor<128x128xbf16>
//     ...
//   }
// }
struct CastCommutationPattern : OpRewritePattern<mlir::tensor::ExtractSliceOp> {
    using OpRewritePattern<mlir::tensor::ExtractSliceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::tensor::ExtractSliceOp op, PatternRewriter &rewriter) const override {
        // Check if there is a castOp that precedes ExtractSliceOp
        auto castOp = op.getSource().getDefiningOp<CastOp>();
        if (!castOp)
            return failure();

        // Get the type and shape pre extract slice
        auto origResultType = op.getResultType(); // get the result type after cast
        auto preCastElemType = mlir::cast<mlir::RankedTensorType>(castOp.getInput().getType()).getElementType();
        // Form a tensor with the orig result type shape and elem type
        auto preCastSliceType = mlir::RankedTensorType::get(origResultType.getShape(), preCastElemType);

        // Slice the cast's input at the same offset/sizes/strides as the ExtractSliceOp
        auto newSlice = mlir::tensor::ExtractSliceOp::create(rewriter, op.getLoc(), preCastSliceType, 
                                                           castOp.getInput(), op.getMixedOffsets(),
                                                           op.getMixedSizes(), op.getMixedStrides());

        // Cast just the small slice, replacing the original (full-tensor slice)
        // Original cast get erased with DCE
        rewriter.replaceOpWithNewOp<CastOp>(op, origResultType, newSlice);
        return success();
    }
};

void populateCastCommutationPatterns(RewritePatternSet &patterns) {
    patterns.add<CastCommutationPattern>(patterns.getContext());
}
} // namespace mlir::seki
