#ifndef SEKI_HW_ATTRS_H
#define SEKI_HW_ATTRS_H
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "seki_hw/SekiHwDialect.h"

#include "seki_hw/SekiHwEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "seki_hw/SekiHwAttrs.h.inc"
#endif // SEKI_HW_ATTRS_H
