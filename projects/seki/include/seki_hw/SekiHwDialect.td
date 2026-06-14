#ifndef SEKI_HW_DIALECT_TD
#define SEKI_HW_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===-----------------------------------------------===//
// Seki HW Dialect
//===-----------------------------------------------===//
def SekiHwDialect : Dialect {
    let name = "seki_hw";
    let cppNamespace = "::mlir::seki_hw";
    let summary = "Seki hardware ISA dialect - tile-level RISC ops mapping to hardware instructions";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

//===-----------------------------------------------===//
// Type base class
//===-----------------------------------------------===//
class SekiHwType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SekiHwDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

//===-----------------------------------------------===//
// Attribute base class
//===-----------------------------------------------===//
class SekiHwAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SekiHwDialect, name, traits> {
    let mnemonic = attrMnemonic;
}
#endif
