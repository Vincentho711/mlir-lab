#include "seki/SekiOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES // Emit method bodies like build(), print(), ...
#include "seki/SekiOps.cpp.inc"

namespace mlir::seki {

mlir::LogicalResult ReluOp::verify() {
    auto inputType = mlir::cast<mlir::TensorType>(getInput().getType());
    if (!mlir::isa<mlir::FloatType>(inputType.getElementType()))
        return emitOpError("requires floating-point element type, got ")
               << inputType.getElementType();
    return mlir::success();
}

// Tells the framework that C (operand index 2) is the destination for C_out = A * B + C_in
mlir::MutableOperandRange MatmulOp::getDpsInitsMutable() {
    return MutableOperandRange(getOperation(), /*start=*/2, /*length=*/1);
}

mlir::LogicalResult MatmulOp::verify() {
    auto A = mlir::cast<mlir::RankedTensorType>(getA().getType());
    auto B = mlir::cast<mlir::RankedTensorType>(getB().getType());
    auto C = mlir::cast<mlir::RankedTensorType>(getC().getType());

    // Check 2-D dimension of all inputs
    if (A.getRank() != 2)
        return emitOpError("A must be 2-D, got rank ") << A.getRank();
    if (B.getRank() != 2)
        return emitOpError("B must be 2-D, got rank ") << B.getRank();
    if (C.getRank() != 2)
        return emitOpError("C must be 2-D, got rank ") << C.getRank();

    // Check A, B and C shape compatability for matmul
    if (A.getDimSize(1) != B.getDimSize(0))
        return emitOpError("inner dimensions must match: A columns (")
            << A.getDimSize(1) << ") != B rows (" << B.getDimSize(0) << ")";
    if (A.getDimSize(0) != C.getDimSize(0))
        return emitOpError("M dimensions mismatch: A rows (")
            << A.getDimSize(0) << ") != C rows (" << B.getDimSize(0) << ")";
    if (B.getDimSize(1) != C.getDimSize(1))
        return emitOpError("M dimensions mismatch: B columns (")
            << B.getDimSize(1) << ") != C columns (" << C.getDimSize(1) << ")";

    // Type checks
    if (!mlir::isa<mlir::FloatType>(A.getElementType()))
        return emitOpError("A must have floating-point element type, got ")
            << A.getElementType();
    if (A.getElementType() != B.getElementType())
        return emitOpError("A and B element types must match: ")
            << A.getElementType() << " vs " << B.getElementType();
    if (A.getElementType() != C.getElementType())
        return emitOpError("accumulator C element types must match A/B: ")
            << C.getElementType() << " vs " << A.getElementType();

    return mlir::success();
}

} // namespace mlir::seki
