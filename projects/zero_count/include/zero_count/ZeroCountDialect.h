#ifndef ZERO_COUNT_DIALECT_H
#define ZERO_COUNT_DIALECT_H
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

// Include the generated dialect class
#include "zero_count/ZeroCountDialect.h.inc"

// Include the generation op classes (CountZerosOp)
// GET_OP_CLASSES tells the .inc file to emit the full class bodies, not just forward
// declarations at the top of the file.
#define GET_OP_CLASSES
#include "zero_count/ZeroCountOps.h.inc"

#endif
