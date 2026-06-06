#ifndef SEKI_OPS_H
#define SEKI_OPS_H
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "seki/SekiDialect.h"
#define GET_OP_CLASSES // Emit the full class definition
#include "seki/SekiOps.h.inc"
#endif // SEKI_OPS_H
