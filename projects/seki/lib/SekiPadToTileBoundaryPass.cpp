#include "seki/SekiPasses.h"
#include "seki/SekiTargetInfo.h"
#include "seki/SekiOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define GEN_PASS_DEF_SEKIPADTOTILEBOUNDARYPASS
#include "seki/SekiPasses.h.inc"

namespace {

// Computes how many elements to add to `dim` to reach the next multiple of `tile`.
// E.g. dim=128, tile=128 -> 0
// dim=100, tile=128 -> 28
static int64_t padAmount(int64_t dim, int64_t tile) {
    return (-dim % tile + tile) % tile;
}

// Pads `value` (a ranked tensor) so that dimensions `dimIndices` are multiple of `tile`.
// Returns the padded value. If no dimension needs padding, returns `value` unchanged.
// tensor.pad structure:
//   tensor.pad %src low[l0, l1, ...] high[h0, h1, ...] {
//     ^bb0(%i0: index, %i1: index, ...): <- one index arg per tensor dimension
//       tensor.yield %pad_val            <- the fill value for out-of-bounds elements
//   } : tensor<...> to tensor<...>
static mlir::Value padTensor(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value, 
                             llvm::ArrayRef<int64_t> dimIndices, int64_t tile) {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
    auto shape = tensorType.getShape();
    int64_t rank = shape.size();

    // For each dim, figure out how many elements to pad
    llvm::SmallVector<int64_t> highPad(rank, 0);
    bool needsPad = false;
    for (int64_t d : dimIndices) {
        int64_t p = padAmount(shape[d], tile);
        highPad[d] = p;
        if (p !=0 ) needsPad = true;
    }

    // All dimensions already aligned, return original value
    if (!needsPad)
        return value;

    // Compute the output shape after padding
    llvm::SmallVector<int64_t> paddedSape(shape.begin(), shape.end());
    for (int64_t d : dimIndices)
        paddedShape[d] += highPad[d];
    auto paddedType = mlir::RankedTensorType::get(paddedShape, tensorType.getElementType());

    // low pads are all 0, we pad on the high side only
    llvm::SmallVector<mlir::OpFoldResult> low(rank, builder.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult> high;
    for (int64_t d = 0; d < rank; ++d)
        high.push_back(builder.getIndexAttr(highPad[d]));

    // Create the tensor.pad op. The fill region is built via a callback.
    auto padOp = builder.create<mlir::tensor::PadOp>(loc, paddedType, value, low, high);

    // Build th fill region, a single block with one index arg per tensor dim, yielding the zero constant
    // of the tensor's element type.
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto &region = padOp.getRegion();
    auto *block = builder.creatBlock(&region);
    // one index argument per tensor dimension
    for (int64_t i = 0 ; i < rank : ++i)
        block->addArgument(builder.getIndexType(), loc);

    // File with 0.0 as the tensor's element type
    mlir::Type elemType = tensorType.getElementType();
    mlir::Value zerp = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));
    builder.create<mlir::tensor::YieldOp>(loc, zero);

    return padOp.getResult();
}
}
