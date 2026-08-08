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
    SekiPadTypeConverter(int64_t tileRows, int64_t tileCols) : tileRows_(tileRows),
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

        // Source materialisation: padded -> original.
        // Fires at func return. Inserts tensor.extract_slice to strip padding back to ABI shape 
        addSourceMaterialization([](mlir::OpBuilder &builder, mlir::RankedTensorType resultType, 
                                    mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value{
            if (inputs.size() != 1)
                return mlir::Value{};
            return buildSlice(builder, loc, inputs[0], resultType);
        });

        // Target materialisation: orignal -> padded.
        // Fires at function argument use sites
        // Inserts: tensor.pad to widen the original tensor to the tile-aligned shape.
        addTargetMaterialization([](mlir::OpBuilder &builder, mlir::RankedTensorType resultType,
                                    mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
            if (inputs.size() != 1)
                return mlir::Value{};
            return buildPad(builder, loc, inputs[0], resultType);
        });
    }
private:
    int64_t tileRows_;
    int64_t tileCols_;

    // Ceiling-division alignment: smallest multiple of tile >= dim.
    static int64_t alignTo(int64_t dim, int64_t tile) {
        return ((dim + tile - 1) / tile) * tile;
    }

    // Builds tensor.pad from src to targetType with zero-fill, high side only.
    static mlir::Value buildPad(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value src, mlir::RankedTensorType targetType) {
        auto srcType = mlir::cast<mlir::RankedTensorType>(src.getType());
        int64_t rank = srcType.getRank();
        // low[i] = 0 for all dims
        // high[i]  = targetDim - srcDim
        llvm::SmallVector<mlir::OpFoldResult> low(rank, builder.getIndexAttr(0));
        llvm::SmallVector<mlir::OpFoldResult> high;
        for (int64_t i = 0; i < rank; ++i)
            high.push_back(builder.getIndexAttr(targetType.getDimSize(i) - srcType.getDimSize(i)));
        auto padOp = mlir::tensor::PadOp::create(builder, loc, targetType, src, low, high);
        // tensor.pad requires a fill region
        // InsertionGuard saves the builder's current insertion point and restores it when
        // guard goes out of scope. 
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block *block = builder.createBlock(&padOp.getRegion());
        for (int64_t i = 0; i < rank; ++i)
             block->addArgument(builder.getIndexType(), loc);

        // Fill Zeros in the tensor's element type
        mlir::Type elemType = srcType.getElementType();
        mlir::Value zero;
        if (mlir::isa<mlir::FloatType>(elemType))
            zero = mlir::arith::ConstantOp::create(builder, loc, builder.getFloatAttr(elemType, 0.0));
        else
             zero = mlir::arith::ConstantOp::create(builder, loc, builder.getIntegerAttr(elemType, 0));
        mlir::tensor::YieldOp::create(builder, loc, zero);
        return padOp.getResult();
    }

    // Builds tensor.extract_slice that strips padding, returning resultType from src
    static mlir::Value buildSlice(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value src,
                                  mlir::RankedTensorType resultType) {
        int64_t rank = resultType.getRank();
        // offsets[i] = 0, the valid data lives at the start of every dimensions.
        // sizes[i] = resultType.getDimSize(i), the original unpadded extent.
        // strides[i] = 1, no skipping, contiguous slice.
        llvm::SmallVector<mlir::OpFoldResult> offsets(rank, builder.getIndexAttr(0));
        llvm::SmallVector<mlir::OpFoldResult> sizes;
        for (int64_t i = 0; i < rank; ++i)
            sizes.push_back(builder.getIndexAttr(resultType.getDimSize(i)));
        llvm::SmallVector<mlir::OpFoldResult> strides(rank, builder.getIndexAttr(1));

        return mlir::tensor::ExtractSliceOp::create(
            builder, loc, resultType, src, offsets, sizes, strides
        );
    }

};

// Generic conversion pattern for seki dps whose only change is that their result types
// need to be widened to tile-aligned shapes.
template<typename OpTy>
struct SekiTypeUpdatePattern : mlir::OpConversionPattern<OpTy> {
    using mlir::OpConversionPattern<OpTy>::OpConversionPattern;
    // Example 1:
    // Before this IR pattern fires:
    // %X : tensor<100x500xbf16>
    // %W1 : tensor<500x256xbf16>
    // %acc1: tensor<100x256xf32>
    // %H = seki.matmul ins(%X, %W1)  outs(acc1) -> tensor<100x256xf32>
    // %Hact = seki.ew_unary<relu> %H : tensor<100x256xf32>
    // return %Hact : tensor<100x256xf32> 
    // After this IR pattern fires:
    // %X_pad = tensor.pad %X ... : tensor<128x512xbf16>
    // %W_pad = tensor.pad %X ... : tensor<512x256xbf16>
    // %acc1_pad = tensor.pad %acc1 ... : tensor<128x256xf32>
    // %H = seki.matmul ins(%X_pad, %W1_pad) outs(%acc2_pad) -> tensor<128x256xf32> 
    // %Hact = seki.ew_unary<relu> %H : tensor<100x256xf32>
    // %strip = tensor.extract_slice %Hact[0,0][100, 256][1, 1]: tensor<100x256xf32>
    // return %strip : tensor<100x256xf32> 
    mlir::LogicalResult matchAndRewrite(
        OpTy op,
        typename mlir::OpConversionPattern<OpTy>::OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        // Ask the TypeConverter what the result types should be after conversion.
        // convertTypes walks op->getResultTypes() and calls addConversion for each.
        // E.g. tensor<100x256xf32> this returns tensor>128x256xf32> 
        llvm::SmallVector<mlir::Type> newTypes;
        if (mlir::failed(this->typeConverter->convertTypes(op->getResultTypes(), newTypes)))
            return mlir::failure();
        // Recreate the op with:
        //     newTypes              - the padded result types
        //     adaptor.getOperands() - operands already materialised to padded types
        //     op->getAttrs()        - all attributes preserved unchanged
        // E.g. Creates: %H = seki.matmul ins(%X_pad, %W1_pad, %acc1_pad) outs(%acc2_pad) -> tensor<128x256xf32>
        // Replaces all uses of old %H (tensor<100x256xf32)  with new %H (tensor<128x256xf32>)
        rewriter.replaceOpWithNewOp<OpTy>(op, newTypes, adaptor.getOperands(), op->getAttrs());
        return mlir::success();
    }

};

// Dedicated pattern for tensor.empty
// tensor.empty creates uninitialised buffer which is the standard way to allocate
// DPS `outs` operand for seki.matmul / seki.batch_matmul.
// tensor.empty has no generate (TypeRange, ValueRange, NamedAttrList) builder
// Its builder signature is (ArrayRef<int64_t> staticShape, Type elementType, ValueRange dynamicSizes).
// we have a special conversion pattern for it
struct EmptyOpConversionPattern : mlir::OpConversionPattern<mlir::tensor::EmptyOp> {
    using mlir::OpConversionPattern<mlir::tensor::EmptyOp>::OpConversionPattern;
    mlir::LogicalResult matchAndRewrite(mlir::tensor::EmptyOp op, mlir::OpConversionPattern<mlir::tensor::EmptyOp>::OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        // convertTypes applies the addConversion rule for a single type.
        // E.g. tensor<100x256xf32> -> tensor<128x256xf32> 
        mlir::Type newType = this->typeConverter->convertType(op.getType());
        if (!newType)
            return mlir::failure();
        auto paddedType = mlir::cast<mlir::RankedTensorType>(newType);
        // Replace with a tensor empty of the padded shape, no operands, no attributes.
        rewriter.replaceOpWithNewOp<mlir::tensor::EmptyOp>(op, paddedType.getShape(), paddedType.getElementType());
        return mlir::success();
    }
};

struct SekiPadToTileBoundaryPass : impl::SekiPadToTileBoundaryPassBase<SekiPadToTileBoundaryPass> {
    using SekiPadToTileBoundaryPassBase::SekiPadToTileBoundaryPassBase;

    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();
        if (!mod->getAttrOfType<mlir::seki::SekiTargetAttr>("seki.target"))
            return;
        mlir::seki::SekiTargetInfo info(mod);
        if (!info.hasMACArray())
            return;
        int64_t tileRows = info.getMACRows();
        int64_t tileCols = info.getMACCols();

        SekiPadTypeConverter typeConverter(tileRows, tileCols);

        // Describe what IR is legal after conversion.
        // ConversionTarget has 3 possible legality states for any op:
        // 1. Always legal, ops that are registered with addLegalOp<>, framework
        // never tries to convert them.
        // 2. Always illegal, ops that are registered with addIllegalOp<>, framework always
        // tries to convert them, false if no pattern matches.
        // Dynamically legal, legality is decidef at runtime by calling a predicate
        // on the actual op instance. It returns true when typeConverter.isLegal(op) returns
        // true due to every one of its operands and result is already a legal type,
        // i.e. it is already tile-aligned after the converter's addConversion() rule has
        // been applied as identify. If isLegal() returns false, that means tensor is not
        // tile-aligned and SekiTypeUpdatePattern fires to update it.
        mlir::ConversionTarget target(getContext());
        target.addLegalOp<mlir::ModuleOp, mlir::func::FuncOp>();
        // func.return needs its own legality predice, not typeConverter.isLegal().
        // typeConverter.isLegal() would return true for tile aligned shapes, so 
        // it would never trigger populateReturnOpTypeConversionPattern. 
        // Instead we check whether the return operand type match the parent function's declared 
        // result type. If not, operand is padded, function says original, the return is illegal 
        // and the pattern fires to insert tensor.extract_slice. 
        target.addDynamicallyLegalOp<mlir::func::ReturnOp>([](mlir::func::ReturnOp op){
            auto func = op->getParentOfType<mlir::func::FuncOp>();
            return func.getResultTypes() == op.getOperandTypes();
        });
        // reshape op is marked legal as the converter inserts tensor.pad at
        // at its consumer edge if the reshape op is not tile-aligned.
        target.addLegalOp<mlir::seki::ReshapeOp>();
        // Ops inserted by the framework's materialisation step is marked legal cause
        // they have mixed types. e.g. tensor.pad takes an unpadded source and produces
        // a padded result, so isLegal() would return false, and framework will try to
        // convert them even though they were just inserted by the materialisation step.
        target.addLegalOp<mlir::tensor::PadOp, mlir::tensor::ExtractSliceOp, mlir::tensor::YieldOp,
                          mlir::arith::ConstantOp>();
        // For all other remaining ops, legality is decidef at runtime by calling predicate on
        // the actual op.
        target.markUnknownOpDynamicallyLegal([&](mlir::Operation *op) { return typeConverter.isLegal(op); });

        // Populate conversion patterns with op that needs to be converted
        // Mainly ops that operate with tensors.
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<SekiTypeUpdatePattern<mlir::seki::MatmulOp>,
                     SekiTypeUpdatePattern<mlir::seki::BatchMatmulOp>,
                     SekiTypeUpdatePattern<mlir::seki::EwUnaryOp>,
                     SekiTypeUpdatePattern<mlir::seki::EwBinaryOp>,
                     SekiTypeUpdatePattern<mlir::seki::CastOp>,
                     EmptyOpConversionPattern>(typeConverter, &getContext());
        // Adds a pattern that rewires func.return operands to the original function return types.
        // triggering target materialisation (tensor.extract_slice) for any padded values being returned.
        mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);

        // Freeze the patterns so it is immutable.
        mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
        // Apply the conversion per function
        mlir::WalkResult walkResult = mod.walk([&](mlir::func::FuncOp func) -> mlir::WalkResult {
            if (mlir::failed(mlir::applyPartialConversion(func, target, frozenPatterns)))
                return mlir::WalkResult::interrupt();
            return mlir::WalkResult::advance();
        });
        if (walkResult.wasInterrupted())
            signalPassFailure();
  }

};
} // namespace
