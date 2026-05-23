#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "zero_count/ZeroCountDialect.h"

namespace {

struct CountZeroLowering : public mlir::OpRewritePattern<mlir::zero_count::CountZerosOp> {
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::zero_count::CountZerosOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::Location loc = op.getLoc();
        // Get the raw value to count zeros for
        mlir::Value input = op.getInput();

        // This is the lowering process
        // Create a operation which creates -1 in i32, this is -xFFFFFFFF - all 32 bits set
        auto negOne = mlir::arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(-1));
        // XOR with all-ones flips every bit, from 1 to 0.
        auto flipped = mlir::arith::XOrIOp::create(rewriter, loc, input, negOne);
        // Count the number of 1-bits in the flipped value, which equates to the number of 0-bits in the original
        auto zeros = mlir::math::CtPopOp::create(rewriter, loc, flipped);

        // Replace the original op with the number of zeros
        rewriter.replaceOp(op, zeros);
        return mlir::success();
    }
};

struct LowerZeroCountPass : public mlir::PassWrapper<LowerZeroCountPass, mlir::OperationPass<mlir::func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerZeroCountPass)

    mlir::StringRef getArgument()   const override { return "lower-zero-count"; }
    mlir::StringRef getDescription() const override { return "Lower zero_count to arith + math" ; }

    // If your pass introduces ops from a dialect that was not already present in the input IR, declare that dialect in getDependentDialects
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::arith::ArithDialect,
                        mlir::math::MathDialect>();
    }

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<CountZeroLowering>(&getContext());
        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};
}

namespace mlir::zero_count {
void registerZeroCountPasses() {
    mlir::PassRegistration<LowerZeroCountPass>();
}
}
