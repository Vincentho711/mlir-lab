#include "seki/SekiOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES // Emit method bodies like build(), print(), ...
#include "seki/SekiOps.cpp.inc"

namespace mlir::seki {

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

    // A: [M, K]
    // B: [K, N] normally, or [N, K] with tranpose_b
    // C: [M, N]
    // Check A, B and C shape compatability for matmul
    bool transB = getTransposeB().has_value();
    int64_t M = A.getDimSize(0), K = A.getDimSize(1);
    int64_t Krow = transB ? B.getDimSize(1) : B.getDimSize(0);
    int64_t N = transB ? B.getDimSize(0) : B.getDimSize(1);
    if (K != Krow)
        return emitOpError("inner dimensions must match: A K (")
            << K << ") != B K (" << Krow << ")";
    if (M != C.getDimSize(0))
        return emitOpError("M dimensions mismatch: A rows (")
            << M << ") != C rows (" << B.getDimSize(0) << ")";
    if (N != C.getDimSize(1))
        return emitOpError("N dimensions mismatch: B columns (")
            << N << ") != C columns (" << C.getDimSize(1) << ")";

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

// Tells the framework that C (operand index 2) is the destination for C_out = A * B + C_in
mlir::MutableOperandRange BatchMatmulOp::getDpsInitsMutable() {
    return MutableOperandRange(getOperation(), /*start=*/2, /*length=*/1);
}

mlir::LogicalResult ReduceOp::verify() {
    auto in = mlir::cast<mlir::RankedTensorType>(getInput().getType());
    auto out = mlir::cast<mlir::RankedTensorType>(getOutput().getType());
    int64_t rank = in.getRank();
    int64_t axis = getAxis();

    if (axis < 0 || axis >= rank)
        return emitOpError("axis ") << axis << " out of range for rank-" << rank << " input";
    if (out.getRank() != rank - 1)
        return emitOpError("output rank must be input rank minus 1 (")
               << (rank - 1) << "), got " << out.getRank();
    for (int64_t i = 0, j = 0; i < rank; ++i) {
        if (i == axis) continue; // i walks the input dimensions, j walks the output dimensions - and j only advances when i is not the axis being reduced.
        // j lags i by 1 after the axis, i=j before the axis
        // Check that the dimension of input[i] and output[j] is the same
        if (in.getDimSize(i) != out.getDimSize(j))
            return emitOpError("output dim ") << j << " (" << out.getDimSize(j)
                << ") must match input dim " << i << " (" << in.getDimSize(i) << ")";
        ++j;
    }
    return mlir::success();
}

mlir::LogicalResult BatchMatmulOp::verify() {
    auto A = mlir::cast<mlir::RankedTensorType>(getA().getType());
    auto B = mlir::cast<mlir::RankedTensorType>(getB().getType());
    auto C = mlir::cast<mlir::RankedTensorType>(getC().getType());

    if (A.getRank() != 3) return emitOpError("A must be 3-D, got rank ") << A.getRank();
    if (B.getRank() != 3) return emitOpError("B must be 3-D, got rank ") << B.getRank();
    if (C.getRank() != 3) return emitOpError("C must be 3-D, got rank ") << C.getRank();

    // Check if B is transposed, get the right dim
    bool transB = getTransposeB().has_value();
    int64_t batchA = A.getDimSize(0), M = A.getDimSize(1), K = A.getDimSize(2);
    int64_t batchB = B.getDimSize(0);
    int64_t Krow   = transB ? B.getDimSize(2) : B.getDimSize(1);
    int64_t N      = transB ? B.getDimSize(1) : B.getDimSize(2);

    if (batchA != batchB) return emitOpError("batch mismatch: A=") << batchA << " B=" << batchB;
    if (K != Krow)        return emitOpError("inner dimension mismatch: A K=") << K << " B K=" << Krow;
    if (batchA != C.getDimSize(0)) return emitOpError("C batch must equal A batch");
    if (M != C.getDimSize(1))      return emitOpError("C rows must equal A rows");
    if (N != C.getDimSize(2))      return emitOpError("C cols must equal B cols");

    if (!mlir::isa<mlir::FloatType>(A.getElementType()))
        return emitOpError("requires floating-point element type");
    if (A.getElementType() != B.getElementType())
        return emitOpError("A and B element types must match");
    if (A.getElementType() != C.getElementType())
        return emitOpError("C element type must match A/B");
    return mlir::success();
}


mlir::LogicalResult BroadcastOp::verify() {
    auto in  = mlir::cast<mlir::RankedTensorType>(getInput().getType());
    auto out = mlir::cast<mlir::RankedTensorType>(getOutput().getType());
    auto dims = getBroadcastDims();

    if (static_cast<int64_t>(dims.size()) != in.getRank())
        return emitOpError("broadcast_dims size (") << dims.size()
               << ") must equal input rank (" << in.getRank() << ")";
    if (out.getRank() < in.getRank())
        return emitOpError("output rank must be >= input rank");
    for (auto [i, d] : llvm::enumerate(dims))
        if (d < 0 || d >= out.getRank())
            return emitOpError("broadcast_dims[") << i << "]=" << d
                   << " out of range for output rank " << out.getRank();
    return mlir::success();
}

mlir::LogicalResult TransposeOp::verify() {
    auto in   = mlir::cast<mlir::RankedTensorType>(getInput().getType());
    auto out  = mlir::cast<mlir::RankedTensorType>(getOutput().getType());
    // E.g. tensor<A x B x C> transpose to tensor<C x A x B> with permutation = [2, 0, 1]
    auto perm = getPermutation();
    int64_t rank = in.getRank();

    if (static_cast<int64_t>(perm.size()) != rank)
        return emitOpError("permutation size must equal input rank");
    if (out.getRank() != rank)
        return emitOpError("output rank must equal input rank");

    llvm::SmallVector<bool> seen(rank, false);
    // E.g. perm = [2, 0, 1]
    // i = [0, 1, 2], p = [2, 0 ,1]
    for (auto [i, p] : llvm::enumerate(perm)) {
        // Check if p is valid
        if (p < 0 || p >= rank)
            return emitOpError("permutation[") << i << "]=" << p << " out of range";
        // Check if there are duplicates in p
        if (seen[p])
            return emitOpError("duplicate value ") << p << " in permutation";
        seen[p] = true;
    }
    // Check if permutation agrees with output dimensions
    for (int64_t i = 0; i < rank; ++i)
        if (in.getDimSize(perm[i]) != out.getDimSize(i))
            return emitOpError("output dim ") << i << " must be input dim " << perm[i];
    return mlir::success();
}

mlir::LogicalResult ReshapeOp::verify() {
    // ReshapeOp changes the shape of a tensor without changing the data or element type.
    // The total number of elements must stay the same, only the layout of dimensions changes. 
    auto in = mlir::cast<mlir::RankedTensorType>(getInput().getType());
    auto out = mlir::cast<mlir::RankedTensorType>(getOutput().getType());
    bool staticIn = true, staticOut = true;
    int64_t inElems = 1, outElems = 1;

    if (in.getElementType() != out.getElementType())
        return emitOpError("element type mismatch");
    // Lopp through the dim of in, multiply all of the dim vars together
    // Lopp through the dim of out, multiply all of the dim vars together
    for (int64_t d : in.getShape()) { if (d < 0) staticIn = false; else inElems *= d; }
    for (int64_t d : out.getShape()) { if (d < 0) staticOut = false; else outElems *= d; }
    if (staticIn && staticOut && inElems != outElems)
        return emitOpError("element count mismatch: input=") << inElems << " output=" << outElems;
    return mlir::success();

}

} // namespace mlir::seki
