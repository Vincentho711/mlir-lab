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

// Fuses a matmul and its chain of single-use elementwise consumers into one
// seki.fused_computation region. Grows by 1 op per pattern firing, applyPatternsGreedily's
// fixed point handles chain of any length.
struct EpilogueFusionPattern : OpInterfaceRewritePattern<SekiEpilogueFusableOpInterface> {
    using OpInterfaceRewritePattern<SekiEpilogueFusableOpInterface>::OpInterfaceRewritePattern;

    LogicalResult matchAndRewrite(SekiEpilogueFusableOpInterface op, PatternRewriter &rewriter) const override {
        // Guard: never fuse an op that's already inside a fused_computation region.
        if (op->getParentOfType<FusedComputationOp>())
            return failure();
        // Exactly one operand must be the chain predecesser. All other operands are "extra",
        // new values the group must additionally cosume (e.g. the bias tensor feeding a binary add).
        Value chainOperand;
        llvm::SmallVector<Value> extraOperands;
        for (Value operand : op->getOperands()) {
            if (isChainOperand(operand)) {
                if (chainOperand)
                    return failure(); // ambiguous: if there is already a chainOperand from an already-fused chain, we don't process the next.
                chainOperand = operand;
            } else {
                extraOperands.push_back(operand);
            }
        }
        if (!chainOperand)
            return failure();

        // Classify the chain's predecesser, from a matmul or absorb into an existing group.
        // Both branches build the same way.
        llvm::SmallVector<Value> groupInputs;
        llvm::SmallVector<Operation *> bodyOps;

        Operation *predDefOp = chainOperand.getDefiningOp();
        if (auto matmulOp = mlir::dyn_cast<MatmulOp>(predDefOp)) {
            // No existing group, the matmul itself becomes the first body op of the fused_computation.
            groupInputs = {matmulOp.getA(), matmulOp.getB(), matmulOp.getC()};
            bodyOps.push_back(matmulOp);
        } else if (auto groupOp = mlir::dyn_cast<FusedComputationOp>(predDefOp)) {
            // Absorb, there is already a return from a fused_computation op, hence we can reuse.
            groupInputs.assign(groupOp.getInputs().begin(), groupOp.getInputs().end());
            for (Operation &bodyOp : groupOp.getBody().front().without_terminator()) // without_terminator() iterates every op in the block except the last one.
                bodyOps.push_back(&bodyOp);
        } else {
            // isChainOperand should only permit these 2 case, so this is unreachable.
            return failure();
        }

        // Keep track of the number of existing ops in the fused body
        size_t inheritedCount = bodyOps.size();

        // Resolve each extra operand, and fold them into groupInputs.
        // Each operand is either produced by a single-use seki.broadcast -> absorb the broadcast into the body too, and use its input (the raw bias) as group input.
        // Or each oeprand is anything else -> pass it through unchanged as a new group input.
        for (Value extra: extraOperands) {
            if (auto broadcastOp = extra.getDefiningOp<BroadcastOp>(); broadcastOp && extra.hasOneUse()) {
                bodyOps.push_back(broadcastOp);
                groupInputs.push_back(broadcastOp.getInput());
            } else {
                groupInputs.push_back(extra);
            }
        }
        // This op iteself joins the group as the last body op.
        bodyOps.push_back(op);

        // BUild the new group: a fresh FusedComputationOp whose block argumnts corespond 1:1 with groupInputs, containing
        // every body op in order, terminated by seki.yield. 
        llvm::SmallVector<Type> inputTypes;
        for (Value v : groupInputs)
            inputTypes.push_back(v.getType());

        auto fused = FusedComputationOp::create(
            rewriter, op->getLoc(), op->getResult(0).getType(), groupInputs
        );

        Block *block = rewriter.createBlock(&fused.getBody());
        for (Type t : inputTypes)
            block->addArgument(t, op->getLoc());

        // Seed the mapping with outer values -> new block args.
        IRMapping mapping;
        // IRMapping: outer values (%A, %B, %C) -> block arguments (%a, %b, %c).
        // Cloning the matmul with this mapping rewires its operands automatically
        // %A -> %a, %B -> %b, %C -> %c
        for (auto [outer, arg] : llvm::zip(groupInputs, block->getArguments()))
            mapping.map(outer, arg);

        // Absort case onlu, the reused body ops reference the old block's arugments, a different set of
        // Value objects from groupInputs. Map them positionally onto the same new block args.
        if (auto groupOp = mlir::dyn_cast<FusedComputationOp>(predDefOp)) {
            // Get the existing block
            Block &oldBlock = groupOp.getBody().front();
            for (auto [oldArg, newArg] : llvm::zip(oldBlock.getArguments(), block->getArguments()))
                mapping.map(oldArg, newArg);
        }

        rewriter.setInsertionPointToStart(block);
        Operation *lastCloned = nullptr;
        for (size_t i = 0; i < bodyOps.size(); ++i) {
            lastCloned = rewriter.clone(*bodyOps[i], mapping);
            // Map this op's old result to its clone's result so the next
            // body op (which may consume it) picks up the in-region value.
            mapping.map(bodyOps[i]->getResult(0), lastCloned->getResult(0));
            if (i + 1 == inheritedCount) {
                // chainOperand is the predecesser's result as seen from outside the group.
                mapping.map(chainOperand, lastCloned->getResult(0));
            }
        }

        YieldOp::create(rewriter, op->getLoc(), lastCloned->getResult(0));

        // Replace the original op's result with the group's result.
        rewriter.replaceOp(op, fused.getResult());

        // Erase the old predecesser only if it was bare matmul
        if (auto matmulOp = mlir::dyn_cast<MatmulOp>(predDefOp))
            rewriter.eraseOp(matmulOp);

        return success();
    }
private:
    static bool isChainOperand(Value v) {
        if (!v.hasOneUse())
            return false;
        Operation *def = v.getDefiningOp();
        return mlir::isa_and_nonnull<MatmulOp, FusedComputationOp>(def);
    }
};

void populateEpilogueFusionPatterns(RewritePatternSet &patterns) {
    patterns.add<EpilogueFusionPattern>(patterns.getContext());
}

} // namespace mlir::seki
