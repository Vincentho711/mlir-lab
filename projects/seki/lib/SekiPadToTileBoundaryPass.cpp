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

        // Source materialisation, called when it needs to produce a padded value from
        // an original-type value.
        // Files at every function argument use site where a converted op expects the padded type. 
        addSourceMaterialization([](mlir::OpBuilder &builder, mlir::RankedTensorType resultType, 
                                    mlir::ValueRange inputs, mlir::Location loc) -> std::optional<mlir::Value>{
            if (inputs.size() != 1)
                return std::nullopt;
            return buildPad(builder, loc, inputs[0], resultType);
        });

        // Target materialisation, called when it needs to produce an orignal-type value
        // from a padded value, fires at func.return where the function signature expects the original 
        // unpadded type. 
        // Inserts: tensor.extract_slice %padded[0, ...][originalSizes][1, ...]
        addTargetMaterialization([](mlir::OpBuilder &builder, mlir::RankedTensorType resultType,
                                    mlir::ValueRange inputs, mlir::Location loc) -> std::optional<mlir::Value> {
            if (inputs.size() != 1)
                return std::nullopt;
            return buildSlice(builder, loc, inputs[0], resultType);
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
        auto padOp = builder.create<mlir::tensor::PadOp>(loc, targetType, src, low, high);
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
            zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));
        else
             zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(elemType, 0));
        builder.create<mlir::tensor::YieldOp>(loc, zero);
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

        return builder.create<mlir::tensor::ExtractSliceOp>(
            loc, resultType, src, offsets, sizes, strides
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
} // namespace
