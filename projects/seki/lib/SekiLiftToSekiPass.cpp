#include "seki/SekiPasses.h"
#include "seki/SekiDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_SEKILIFTTOSEKIPASS
#include "seki/SekiPasses.h.inc"

void populateSekiLiftingPatterns(mlir::RewritePatternSet &patterns);

namespace {
struct SekiLiftToSekiPass : impl::SekiLiftToSekiPassBase<SekiLiftToSekiPass> {
    using SekiLiftToSekiPassBase::SekiLiftToSekiPassBase;

    void runOnOperation() override {
        mlir::func::FuncOp func = getOperation();
        mlir::MLIRContext *ctx = func.getContext();

        mlir::RewritePatternSet patterns(ctx);
        populateSekiLiftingPatterns(patterns);

        if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns))))
            return signalPassFailure();

        // Any linalg.* or tensor reshape op that survived pattern rewriting was not recognised, 
        // report each one and fail
        bool anyUnrecognised = false;
        func.walk([&](mlir::Operation *op){
            if (llvm::isa<mlir::linalg::LinalgOp>(op) || llvm::isa<mlir::tensor::CollapseShapeOp,
                mlir::tensor::ExpandShapeOp>(op)) {
                op->emitError("unrecognised op cannot be lifted to seki dialect");
                anyUnrecognised = true;
            }
        });
        if (anyUnrecognised)
            signalPassFailure();
    }
};
} // namespace

namespace mlir::seki {
std::unique_ptr<mlir::Pass> createSekiLiftToSekiPass() {
    return std::make_unique<SekiLiftToSekiPass>();
}
} // namespace mlir::seki
