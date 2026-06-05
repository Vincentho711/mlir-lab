#include "seki/SekiOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES // Emit method bodies like build(), print(), ...
#include "seki/SekiOps.cpp.inc"

namespace mlir::seki {

mlir::LogicalResult ReluOp::verify(){
    auto inputType = mlir::cast<mlir::TensorType>(getInput().getType());
    if (!mlir::isa<mlir::FloatType>(inputType.getElementType()))
        return emitOpError("requires floating-point element type, got ")
            << inputType.getElementType();
    return mlir::success();
  }
}
