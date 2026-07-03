#include "seki/SekiOps.h"
#include "seki/SekiAttrs.h"
#include "seki_hw/SekiHwOps.h"
#include "seki_hw/SekiHwAttrs.h"
#include "mlir/Transforms/DialectConversion.h"

// Integer alignment guards: seki::UnaryFn and seki_hw::UnaryFn must share the same values for the enums.
// A future reordering of either enum will cause a compile-time failure here.
static_assert(static_cast<int>(mlir::seki::UnaryFn::relu) == static_cast<int>(mlir::seki_hw::UnaryFn::relu));
static_assert(static_cast<int>(mlir::seki::UnaryFn::exp) == static_cast<int>(mlir::seki_hw::UnaryFn::exp));
static_assert(static_cast<int>(mlir::seki::UnaryFn::log) == static_cast<int>(mlir::seki_hw::UnaryFn::log));
static_assert(static_cast<int>(mlir::seki::UnaryFn::neg) == static_cast<int>(mlir::seki_hw::UnaryFn::neg));
static_assert(static_cast<int>(mlir::seki::UnaryFn::abs) == static_cast<int>(mlir::seki_hw::UnaryFn::abs));

namespace {

struct EwUnaryOpLowering : mlir::OpConversionPattern<mlir::seki::EwUnaryOp> {
    using OpConversionPattern::OpConversionPattern;
    mlir::LogicalResult matchAndRewrite(
        mlir::seki::EwUnaryOp op,
        mlir::seki::EwUnaryOp::Adaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter
    ) const override {
        // op.getResult.getType() returns tensor<128x128xf32>
        // Get the converted result type: tensor<MxMxf32> -> !seki_hw.tile<f32>
        // getTypeConverter->convertType(...) runs that type through the TypeConverter registered in that pass, from SekiLowerToSekiHwPass.cpp
        // Hence convert to !seki_hw.tile<f32>
        mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

        // static_asserrts above gurantee integer value alignment
        auto hwFn = static_cast<mlir::seki_hw::UnaryFn>(op.getFn());
        auto hwFnAttr = mlir::seki_hw::UnaryFnAttr::get(rewriter.getContext(), hwFn);

        // Build the replacement op and replace with the original in one call.
        // replaceOpWithNewOp removes the old and splices in the new one.
        rewriter.replaceOpWithNewOp<mlir::seki_hw::TileEwUnaryOp>(op, resultType, hwFnAttr, adaptor.getInput());
        return mlir::success();
    }
};

} // namespace

void populateSekiToSekiHwConversionPatterns(mlir::TypeConverter &converter, mlir::RewritePatternSet &patterns) {
    patterns.add<EwUnaryOpLowering>(converter, patterns.getContext());
}
