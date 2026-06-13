#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "seki_hw/SekiHwDialect.h"
#include "seki_hw/SekiHwOps.h"
#include "seki_hw/SekiHwTypes.h"
#include "seki_hw/SekiHwAttrs.h"

#include "seki_hw/SekiHwDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "seki_hw/SekiHwTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "seki_hw/SekiHwAttrs.cpp.inc"

namespace mlir::seki_hw {

void SekiHwDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
    #include "seki_hw/SekiHwTypes.cpp.inc"
    >();
    addAttributes<
#define GET_ATTRDEF_LIST
    #include "seki_hw/SekiHwAttrs.cpp.inc"
    >();
    addOperations<
#define GET_OP_LIST
#include "seki_hw/SekiHwOps.cpp.inc"
    >();
}

} // namespace mlir::seki_hw
