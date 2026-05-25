#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "zero_count/ZeroCountDialect.h"

#include "zero_count/ZeroCountDialect.cpp.inc"

// Include the generated op class *definitions*
// GET_OP_CLASSES is required to emit the full bodies, not just forward declarations
#define GET_OP_CLASSES
#include "zero_count/ZeroCountOps.cpp.inc"

// Verify CountZerosInRangeOp to ensure compile time attributes are valid
mlir::LogicalResult mlir::zero_count::CountZerosInRangeOp::verify() {
    int32_t lo = getLo();
    int32_t hi = getHi();

    if (lo < 0)
        return emitOpError("lo must be non-negative, got ") << lo;
    if (hi > 32)
        return emitOpError("hi must be <= 32 for i32, got ") << hi;
    if (lo >= hi)
        return emitOpError("lo (") << lo << ") must be less than hi (" << hi << ")";

    return mlir::success();
}

// Folding of the CountZerosOp
mlir::OpFoldResult mlir::zero_count::CountZerosOp::fold(FoldAdaptor adaptor) {
    // Attribute is a base clase, the specific type here is IntegerAttr (an attribute holding an integer). dyn_cast_or_null attempts to downcast 
    // and returns nullptr if it fails or if the input was already nullptr. So it handles both cases: input not a constant, and input is a constant of the wrong kind.
    auto inputAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getInput()); // adaptor.getInput() -> mlir::Attribute (a compile-time constant, or nullptr if unknown)
    if (!inputAttr) // Check if it is a constant, if not return {} to signal no fold applies
        return {};
    uint32_t val = static_cast<uint32_t>(inputAttr.getValue().getZExtValue()); // IntegerAttr stores its value as an APInt (LLVM arbitrary-precision integer type.). getValue() returns that APInt.
    // .getZExtValue() extracts APInt as a uint64_t by zero-extending it. We then cast to uint32_t because the type is i32. Zero-extension is correct as we want the raw bit pattern. 
    uint32_t zeros = 32 - llvm::popcount(val); // Counts the number of 1-bits in val.

    return mlir::IntegerAttr::get(getResult().getType(), zeros); // getResult() returns the SSA value that this op produces. getType() returns its MLIR type, so in this case i32.
    // We poass this to IntegerAttr::get so the returned constant has the correct type. Without the type, MLIR would not know how wide the integer is.
    // mlir::IntegerAttr constructs a new compile-time integer attribute with the given type and value.
}

// Implement materializeConstant() such that fold() can turn op into a arith.constant
mlir::Operation *mlir::zero_count::ZeroCountDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type, mlir::Location loc) {
    // TypedAttr is MLIR's interface for attributes that carry their own type (IntegerAttr implements it).
    return mlir::arith::ConstantOp::create(builder, loc, mlir::cast<mlir::TypedAttr>(value));
}

// Folding of CountZerosInRangeOp
mlir::OpFoldResult mlir::zero_count::CountZerosInRangeOp::fold(FoldAdaptor adaptor) {
    auto inputAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getInput());
    if (!inputAttr)
        return {};
    int32_t lo = getLo();
    int32_t hi = getHi();

    uint32_t val = static_cast<uint32_t>(inputAttr.getValue().getZExtValue());

    // Computer mask, convert to int64_t to ensure that it will work for when lo = 0 and hi = 32 for 1 << 32 to be valid
    uint32_t mask = static_cast<uint32_t>(((int64_t(1) << (hi - lo)) - 1) << lo);
    // Apply bit mask to isolate interested bits
    uint32_t isolated = val & mask;
    // XOR to flip 0 to 1 such that we can use popcount() to count 1s
    uint32_t flipped = isolated ^ mask;
    uint32_t zeros = llvm::popcount(flipped);

    return mlir::IntegerAttr::get(getResult().getType(), zeros);
}

// Register every op that belongs to this dialect with addOperations<>()
void mlir::zero_count::ZeroCountDialect::initialize() {
    addOperations<
        mlir::zero_count::CountZerosOp,
        mlir::zero_count::CountZerosInRangeOp
    >();
}
