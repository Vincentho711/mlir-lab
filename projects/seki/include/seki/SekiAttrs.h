#ifndef SEKI_ATTRS_H
#define SEKI_ATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "seki/SekiDialect.h"

#include "seki/SekiEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "seki/SekiAttrs.h.inc"
#endif // SEKI_ATTRS_H
