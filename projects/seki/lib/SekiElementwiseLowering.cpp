#include "seki/SekiOps.h"
#include "seki_hw/SekiHwOps.h"
#include "seki_hw/SekiHwAttrs.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

struct ReluOpLowering : mlir::OpConversionPattern<mlir::seki::ReluOp> {
    using OpConversionPattern::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::seki::ReluOp op,
        mlir::seki::ReluOp::Adaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        // Get the converted result type: tensor<MxMxf32> -> !seki_hw.tile<f32>
        mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

        // Build the replacement op and replace with the original in one call.
        // replaceOpWithNewOp removes the old and splices in the new one.
        rewriter.replaceOpWithNewOp<mlir::seki_hw::TileEwUnaryOp>(op, resultType, mlir::seki_hw::UnaryFnAttr::get(rewriter.getContext(),
                                                                                                                  mlir::seki_hw::UnaryFn::relu),
                                                                                                                  adaptor.getInput());

        return mlir::success();

    }
};
} // namespace

void populateSekiToSekiHwConversionPatterns(mlir::TypeConverter &converter, mlir::RewritePatternSet &patterns) {
    patterns.add<ReluOpLowering>(converter, patterns.getContext());
}
