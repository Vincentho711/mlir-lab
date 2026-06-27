#include "seki/SekiPasses.h"
#include "seki/SekiDialect.h"
#include "seki_hw/SekiHwDialect.h"
#include "seki_hw/SekiHwTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

#define GEN_PASS_DEF_SEKILOWERTOSEKIHWPASS
#include "seki/SekiPasses.h.inc"

void populateSekiToSekiHwConversionPatterns(mlir::TypeConverter &converter, mlir::RewritePatternSet &patterns);

namespace {
struct SekiLowerToSekiHwPass : impl::SekiLowerToSekiHwPassBase<SekiLowerToSekiHwPass> {
    using SekiLowerToSekiHwPassBase::SekiLowerToSekiHwPassBase;

    void runOnOperation() override {
        // Returns the FuncOp this pass is anchored on, single func we are lowering
        mlir::func::FuncOp func = getOperation();
        mlir::MLIRContext *ctx = func.getContext();

        // Declare the typeconverter to map between tensor and !seki_hw.tile
        mlir::TypeConverter converter;
        // Identity rule, any type not matched by a later rule passes through unchanged.
        // This includes f32, i63, bf16 scalars, etc.
        converter.addConversion([](mlir::Type t) -> mlir::Type {return t; });
        // Tensor conversion: a 2D ranked tensor with a hardware-supported element type maps to !seki_hw.tile<T>.
        //   tensor<128x128xf32>  -> !seki_hw.tile<f32>
        //   tensor<128x128xbf16> -> !seki_hw.tile<bf16>
        // Returning std::nullopt signals "I don't know how to convert this type" - the framework tries next rule (identity rule)
        // We don't check for the fact that tensor<MxNxT> already has M=N=tile dimension of our target here. 
        // Dimension validation belongs to op verifiers and the lowering pattern.
        converter.addConversion([ctx](mlir::RankedTensorType t) -> std::optional<mlir::Type> {
            if (t.getRank() != 2) return std::nullopt; // Reject any tensor that is not 2D
            mlir::Type elem = t.getElementType();
            if (!elem.isF32() && !elem.isBF16() && !elem.isInteger(8) && !elem.isInteger(32))
                return std::nullopt; // Reject any tensor with type that is not supported
            return mlir::seki_hw::TileType::get(ctx, elem);
        });

        // ConversionTarget
        // Declares what the IR must look like after the pass finishes.
        // applyFullConversion will fail if any op violates this contract.
        mlir::ConversionTarget target(*ctx);

        // Everything in seki_hw is a valid output - these ops are the target we are lowering into.
        target.addLegalDialect<mlir::seki_hw::SekiHwDialect>();

        // Every seki op must be gone by the end of the pass. If any seki op has no lowering pattern, applyFullConversion
        // fails with a clear error pointing to the uncoverted op. 
        target.addIllegalDialect<mlir::seki::SekiDialect>();

        // func::FuncOp is legal only if its signature contains no unconverted tyes.
        // A function that still takes tensor<MxNxT> args is illegal - it means that TypeConverter hasn't been applied to
        // that signature yet. 
        // Dynaically legal means legality is checked at convertion time per-op, not statically for all ops of that type
        target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
            return converter.isSignatureLegal(op.getFunctionType());
        });

        // func::ReturnOp is legal only when all its operands have already been converted. A return of tensor<MxNxT> is illegal.
        // A return of !seki_hw.tile<T> is legal.
        target.addDynamicallyLegalOp<mlir::func::ReturnOp>([&](mlir::func::ReturnOp op) {
            return converter.isLegal(op.getOperandTypes());
        });

        // Patterns + applyFullConversion
        mlir::RewritePatternSet patterns(ctx);

        // This built-in MLIR pattern rewrites the FuncOp (func.func) signature itself using the TypeConverter rules.
        // tensor<128x128xf32> args -> !seki_hw.tile<f32> args, and updates the function type accordingly.
        // Without this, the function signature stays as tensors even after body is converted - leaving the FuncOp dynamically illegal. 
        mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, converter);

        // This built-in MLIR pattern rewrites the func.return operands to their converted types.
        // Without it, func.return stays dynamically illegal even after the body ops are all converted.
        mlir::populateReturnOpTypeConversionPattern(patterns, converter);

        // Adds our custom op patterns
        // E.g. seki.relu -> seki_hw.tile_ew_unary<relu>, etc
        populateSekiToSekiHwConversionPatterns(converter, patterns);

        // applyFullConversion is the engine that actually executes:
        // 1. Visit every op in the FuncOp
        // 2. For each op, checks: is it legal according to target?
        // 3. If illegal, finds a matching pattern and applies it
        // 4. After all rewrites, checks: are there any illegal ops remaining? 
        // 5. If yes, hard failure. If no, success.
        if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns))))
            signalPassFailure();
        // E.g.
        // Visit func.func @relu(tensor<128x128xf32>) -> tensor<128x128xf32>
        //   → dynamically illegal (signature has unconverted tensor type)
        //   → apply FunctionOpInterface pattern
        //   → rewrites signature: tensor<128x128xf32> → !seki_hw.tile<f32>

        // Visit seki.relu
        //   → illegal (SekiDialect is fully illegal)
        //   → apply relu ConversionPattern
        //   → replaces with seki_hw.tile_ew_unary<relu>
        //
        // Visit func.return %out : tensor<128x128xf32>
        //   → dynamically illegal (operand is still tensor type)
        //   → framework updates operand to the converted %out : !seki_hw.tile<f32>
        //
        // Final check: any illegal ops remaining? No → pass succeeds.
    }
};
}
