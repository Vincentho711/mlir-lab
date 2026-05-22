#include "zero_count/ZeroCountDialect.h"

#include "zero_count/ZeroCountDialect.cpp.inc"

// Include the generated op class *definitions*
// GET_OP_CLASSES is required to emit the full bodies, not just forward declarations
#define GET_OP_CLASSES
#include "zero_count/ZeroCountOps.cpp.inc"

// Register every op that belongs to this dialect with addOperations<>()
void mlir::zero_count::ZeroCountDialect::initialize() {
    addOperations<
        mlir::zero_count::CountZerosOp
    >();
}
