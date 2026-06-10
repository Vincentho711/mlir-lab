#ifndef SEKI_HW_OPS_H
#define SEKI_HW_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "seki_hw/SekiHwDialect.h"
#include "seki_hw/SekiHwTypes.h"

#define GET_OP_CLASSES
#include "seki_hw/SekiHwOps.h.inc"

#endif // SEKI_HW_OPTS_H
