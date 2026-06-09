#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "seki_hw/SekiHwDialect.h"
#include "seki_hw/SekiHwTypes.h"

#include "seki_hw/SekiHwDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "seki_hw/SekiHwTypes.cpp.inc"

namespace mlir::seki_hw {

void SekiHwDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
  #include "seki_hw/SekiHwTypes.cpp.inc"
  >();
}

} // namespace mlir::seki_hw
