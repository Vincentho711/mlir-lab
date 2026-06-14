#ifndef SEKI_DIALECT_TD
#define SEKI_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===-----------------------------------------------===//
// Seki Dialect
//===-----------------------------------------------===//
def SekiDialect : Dialect {
    let name = "seki";
    let cppNamespace = "::mlir::seki";
    let summary = "Seki ML inference accelerator compiler dialect";
    let useDefaultAttributePrinterParser = 1;
}

#endif // SEKI_DIALECT_TD
