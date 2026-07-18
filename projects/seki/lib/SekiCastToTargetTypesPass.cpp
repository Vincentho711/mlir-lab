#include "seki/SekiPasses.h"
#include "seki/SekiTargetInfo.h"
#include "seki/SekiOps.h"       // CastOp, MatmulOp, BatchMatmulOp
#include "mlir/IR/Builders.h"   // OpBuilder
#include "mlir/IR/BuiltinOps.h" // ModuleOp

#define GEN_PASS_DEF_SEKICASTTOTARGETTYPESPASS
#include "seki/SekiPasses.h.inc"

namespace {

// Returns `input` unchanged when its element type already matches targetElem.
// Otherwise, inserts a seki.cast immediately before parentOp and returns the cast result.
static mlir::Value maybeCast(mlir::OpBuilder &builder, mlir::Operation *parentOp, mlir::Value input,
                             mlir::Type targetElemType, const mlir::seki::SekiTargetInfo &info, bool &failed) {
    // Get the type of the input tensor
    auto tensorType = mlir::cast<mlir::RankedTensorType>(input.getType());
    mlir::Type fromElemType = tensorType.getElementType();

    // Nothing needs to be done here as input type already same as target type
    if (fromElemType == targetElemType)
        return input;
    // If input type is diffferent from target, check whether we can transform with the hardware
    // by query SekiTargetInfo
    if (!info.isCastSupported(fromElemType, targetElemType)) {
        parentOp->emitError()
            << "MAC array requires input element type " << targetElemType
            << " but operand has type " << fromElemType
            << "; no cast path exists on this target";
        failed = true;
        return input;
    }
    // Preserve the tensor shape and change the element type of the original op
    auto castType = mlir::RankedTensorType::get(tensorType.getShape(), targetElemType);
    // Insert a seki.cast op with builder already positioned just before parentOp by the caller.
    return mlir::seki::CastOp::create(builder, parentOp->getLoc(), castType, input);
}

struct SekiCastToTargetTypesPass : impl::SekiCastToTargetTypesPassBase<SekiCastToTargetTypesPass> {
    using SekiCastToTargetTypesPassBase::SekiCastToTargetTypesPassBase;
    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();
        // SekiTargetInfo's constructor calls report_fatal_error if the attribute
        // is absent, so check before constructing it.
        if (!mod->getAttrOfType<mlir::seki::SekiTargetAttr>("seki.target"))
            return;
        mlir::seki::SekiTargetInfo info(mod);

        // Check if the target has a macArray, if not, there is nothing to narrow
        // Return without signalPassFailure() as scalar only target is valid.
        if (!info.hasMACArray())
            return;

        mlir::Type macInputElem = info.macInputType();
        // OpBuilder is a tool for create new ops and insert them into the IR. 2 responsibilities:
        // 1. setInsertionPoint(op), setInsertionPointAfter(op) and setInsertionPointToStart() to anchor on a where to insert the new op.
        // 2. create<OpType>(loc, resultTypes, operands) create the new op at the insertion point.
        mlir::OpBuilder builder(mod.getContext());
        bool failed = false;

        mod.walk([&](mlir::Operation *op) {
            mlir::Value a, b;
            // Only need to cast for MatmulOp and BatchMatulOp
            if (auto m = mlir::dyn_cast<mlir::seki::MatmulOp>(op)) {
                a = m.getA();
                b = m.getB();
            } else if (auto m = mlir::dyn_cast<mlir::seki::BatchMatmulOp>(op)) {
                a = m.getA();
                b = m.getB();
            } else {
                // If not matmul or batch_matmul, simply return to do nothing.
                return;
            }

            // We need to insert a seki.cast<> just before the seki.matmul or seki.batch_matmul op
            builder.setInsertionPoint(op);

            // Check if we can cast A or B into the supported type
            // If so, maybeCast will insert the new seki.cast op at the appropriate place.
            mlir::Value newA = maybeCast(builder, op, a, macInputElem, info, failed);
            mlir::Value newB = maybeCast(builder, op, b, macInputElem, info, failed);

            // After seki.cast has been inserted, we have to change the input type of A and B to the supported type.
            // Rewire  operands 0 (A) and 1 (B) if the type changed.
            // Index from the argument order in SekiOps.td: A = 0, B = 1 and C = 2.
            if (newA != a) op->setOperand(0, newA);
            if (newB != b) op->setOperand(1, newB);

        });

        if (failed) signalPassFailure();
    }

};

} // namespace
