#include "seki/SekiDialect.h"
#include "seki/SekiAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "seki/SekiDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "seki/SekiAttrs.cpp.inc"

namespace mlir::seki {

void SekiDialect::initialize() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "seki/SekiAttrs.cpp.inc"
    >();
}

} // namespace mlir::seki
