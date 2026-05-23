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

// Register every op that belongs to this dialect with addOperations<>()
void mlir::zero_count::ZeroCountDialect::initialize() {
    addOperations<
        mlir::zero_count::CountZerosOp,
        mlir::zero_count::CountZerosInRangeOp
    >();
}
