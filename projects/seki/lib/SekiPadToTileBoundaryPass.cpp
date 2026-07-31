#include "seki/SekiPasses.h"
#include "seki/SekiTargetInfo.h"
#include "seki/SekiOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DEF_SEKIPADTOTILEBOUNDARYPASS
#include "seki/SekiPasses.h.inc"

namespace {

struct SekiPadTypeConverter : mlir::TypeConverter {
    SekiPadTypeConverter(int64_t tileRows, int64_t tileCols) : tileRows_(tileRows)
        tileCols_(tileCols) {
        // Pass all non-tensor types through unchanged, e.g. index, f32 scalars
        addConversion([](mlir::Type t) -> mlir::Type{ return t; });

        // For ranked tensors, pad the last 2 dims to the next tile multiple.
        // Leading dims (batch/order) are left unchanged.
        // TypeConverter::convertTypes tries registered conversions in reverse
        // registration order. The RankedTensorType lambda is registered second, so
        // it is tried first.
        addConversion([this](mlir::RankedTensorType t) -> mlir::RankedTensorType {
            llvm::SmallVector<int64_t> shape(t.getShape());
            int64_t rank = static_cast<int64_t>(shape.size());
            if (rank >= 2)
                shape[rank - 2] = alignTo(shape[rank - 2], tileRows_);
            if (rank >= 1)
                shape[rank - 1] = alignTo(shape[rank - 1], tileCols_);
            return mlir::RankedTensorType::get(shape, t.getElementType());
        });
    }
private:
    int64_t tileRows_;
    int64_t tileCols_;

    // Ceiling-division alignment: smallest multiple of tile >= dim.
    static in64_t alignTo(int64_t dim, int64_t tile) {
        return ((dim + tile - 1) / tile) * tile;
    }

};
} // namespace

struct SekiPadToTileBoundaryPass : impl::SekiPadToTileBoundaryPassBase<SekiPadToTileBoundaryPass> {
    using SekiPadToTileBoundaryPassBase::SekiPadToTileBoundaryPassBase;

    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();
        if (!mod->getAttrOfType<mlir::seki::SekiTargetAttr>("seki.target"))
            return;
        mlir::seki::SekiTargetInfo info(mod);
        if (!info.hasMACArray())
            return;
        int64_t tile = info.getMACRows();
        mlir::OpBuilder builder(mod.getContext());

        mod.walk([&])(mlir::Operation *op) {
            // Only pad matmul-family ops
            if (!mlir::isa<mlir::seki::MatmulOp, mlir::seki::BatchMatmulOp>(op))
                return;
            builder.setInsertionPoint(op);
            mlir::Location loc = op->getLoc();
            
            if (auto matmul = mlir::dyn_cast<mlir::seki::MatmulOp>(op)) {
                // Save the original shapes before any rewiring
                auto origAType = mlir::cast<mlir::RankedTensorType>(matmul.getA().getType());
                auto origCType = mlir::cast<mlir::RankedTensorType>(matmul.getC().getType());
                int64_t origM = origAType.getDimSize(0);  // rows of A = rows of result
                int64_t origN = origCType.getDimSize(1);  // cols of C = cols of result
                // seki.matmul: A is [M, K], B is [K, N], C/result is [M, N]
                // Pad all dims of A, both M and K, as well as all dims of B, both K and N
                mlir::Value newA = padTensor(builder, loc, matmul.getA(), {0, 1}, tile);
                mlir::Value newB = padTensor(builder, loc, matmul.getB(), {0, 1}, tile);

                // Check if we padded, if not, no rewiring is needed
                if (newA == matmul.getA() && newB == matmul.getB())
                    return;

                // Pad C, it must match the padded M and N of A and B.
                mlir::Value newC = padTensor(builder, loc, matmul.getC(), {0, 1}, tile);

                // Rewire the matmul to use padded operands
                op->setOperand(0, newA);
                op->setOperand(0, newB);
                op->setOperand(0, newC);

                // Set result's type to C's result type due to DPS
                auto paddedResultType = mlir::cast<mlir::RankedTensorType>(newC.getType());
                op->getResult(0).setType(paddedResultType);
                
                // Insert tensor.extract_slice after the matmul to strip the M/N padding
                builder.setInsertionPointAfter(op);
                auto origResultType = mlir::cast<mlir::RankedTensorType>(matmul.getResult.getType())
            }
        }
    }

};
}
