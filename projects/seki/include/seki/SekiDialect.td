#ifndef SEKI_DIALECT_TD
#define SIKI_DIALECT_TD

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

//===-----------------------------------------------===//
// Attribute base class
//===-----------------------------------------------===//
class SekiAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SekiDialect, name, traits> {
    let mnemonic = attrMnemonic;
}

//===-----------------------------------------------===//
// Memory subsystem configuration
//===-----------------------------------------------===//
def SkiMemoryConfigAttr : SekiAttr<"SekiMemoryConfig", "memory_config"> {
    let summary = "Seki memory subsystem hardware configuration";
    let parameters = (ins
        "int64_t":$scratchpadBytes,
        "int64_t":$dramBytes,
        "int64_t":$dmaAlignment,
        "int64_t":$numScratchpadBanks
    );
    let assemblyFormat = "`<` struct(params) `>`";
    let genVerifyDecl = 1;
}

//===-----------------------------------------------===//
// Top-level hardware target
//===-----------------------------------------------===//
def SekiTargetAttr : SekiAttr<"SekiTarget", "target"> {
    let summary = "Seki hardware target description attacted to a module";
    let parameters = (ins
        "::mlir::seki::SekiMemoryConfigAttr":$memory
    );
    let assemblyFormat = "`<` struct(params) `>`";
}

#endif // SEKI_DIALECT_TD
