#include "seki/SekiPasses.h"
#include "seki/SekiFusionPatterns.h"
#include "seki/SekiOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_SEKIFUSEOPSPASS
#include "seki/SekiPasses.h.inc"

namespace {

struct SekiFuseOpsPass : impl::SekiFuseOpsPassBase<SekiFuseOpsPass> {
    using SekiFuseOpsPassBase::SekiFuseOpsPassBase;

    void runOnOperation() override {
        mlir::func::FuncOp func = getOperation();
        mlir::RewritePatternSet patterns(&getContext());

        // Fold seki.transpose (B-operand) + seki.matmul -> seki.matmul {transpose_b}
        mlir::seki::populateTransposeMatmulFusionPatterns(patterns);
        // Fuse matmul + single-op epilogue into seki.fused_computation
        mlir::seki::populateSingleOpEpilogueFusionPatterns(patterns);

        // applyPatternsGreedily runs all patterns to a fixed point
        if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns))))
            signalPassFailure();
    }
};
} // namespace
