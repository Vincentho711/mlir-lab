#include "seki/SekiPasses.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define GEN_PASS_DEF_SEKIVERIFYSTATICSHAPESPASS
#include "seki/SekiPasses.h.inc"

namespace {

struct SekiVerifyStaticShapesPass : impl::SekiVerifyStaticShapesPassBase<SekiVerifyStaticShapesPass> {
    void runOnOperation() override {
        mlir::func::FuncOp funcOp = getOperation();
        bool failed = false;

        // Check function argument types
        for (mlir::Type t : funcOp.getArgumentTypes()) {
            if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(t)) {
                if (!shaped.hasStaticShape()) {
                    funcOp.emitError("seki requires static tensor shapes; function argument has dynamic shape: ") << t;
                    failed = true;
                }
            }
        }

        // Visit the result type of every op nested inside the FuncOp, including itself
        funcOp.walk([&](mlir::Operation *op){
            for (mlir::Type t : op->getResultTypes())  {
                if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(t)) {
                    if (!shaped.hasStaticShape()) {
                        op->emitError("seki requires static tensor shapes; op result has dynamic shape: ") << t;
                        failed = true;
                    }
                }
            }

        });

        if (failed)
            signalPassFailure();
    }
};
} // namespace

namespace mlir::seki{
std::unique_ptr<mlir::Pass> createSekiVerifyStaticShapesPass() {
    return std::make_unique<SekiVerifyStaticShapesPass>();
}

} // namespace mlir::seki

