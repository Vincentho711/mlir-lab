#ifndef SEKI_HW_INTERFACES_H
#define SEKI_HW_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "seki_hw/SekiHwDialect.h"

// No need GET_.*_CLASSES, op interface declarations are not macro-gated, the entire class is emitted.
#include "seki_hw/SekiHwInterfaces.h.inc"

#endif // SEKI_HW_INTERFACES_H
