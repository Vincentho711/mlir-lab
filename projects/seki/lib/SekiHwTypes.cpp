#include "seki_hw/SekiHwTypes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::seki_hw {

mlir::LogicalResult TileType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType) {
    if (!elementType.isF32() && !elementType.isBF16() &&
        !elementType.isInteger(8) && !elementType.isInteger(32))
        return emitError() << "tile element type must be f32, bf16, i8 or i32; got "
                           << elementType;
    return mlir::success();
}

mlir::LogicalResult VecType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType,
    int64_t numElements) {
    if (!elementType.isF32() && !elementType.isBF16())
        return emitError() << "vec element type must be f32 or bf16; got "
                           << elementType;
    if (numElements <= 0)
        return emitError() << "vec numElements must be positive; got "
                           << numElements;
    return mlir::success();
}

} // namespace mlir::seki_hw
