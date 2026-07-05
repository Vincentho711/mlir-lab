#ifndef SEKI_ATTRS_TD
#define SEKI_ATTRS_TD

include "seki/SekiDialect.td"
include "mlir/IR/EnumAttr.td"

//===-----------------------------------------------===//
// Attribute base class
//===-----------------------------------------------===//
class SekiAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SekiDialect, name, traits> {
    let mnemonic = attrMnemonic;
}

//===-----------------------------------------------===//
// Memory space enum
//===-----------------------------------------------===//
def Seki_MemorySpace_DRAM       : I32EnumAttrCase<"DRAM",       0, "dram">;
def Seki_MemorySpace_Scratchpad : I32EnumAttrCase<"Scratchpad", 1, "scratchpad">;

def Seki_MemorySpace : I32EnumAttr<"MemorySpace", "seki memory space",
    [Seki_MemorySpace_DRAM, Seki_MemorySpace_Scratchpad]> {
    let cppNamespace = "::mlir::seki";
    // Only generate the C++ enum, we pick the mnemonic and assemblyFormat by specifying our own Seki_MemorySpaceAttr
    let genSpecializedAttr = 0;
}

def Seki_MemorySpaceAttr : EnumAttr<SekiDialect, Seki_MemorySpace, "memory_space"> {
    let assemblyFormat = "`<` $value `>`";
}

//===-----------------------------------------------===//
// UnaryFn — seki.ew_unary<fn>
//===-----------------------------------------------===//
def Seki_UnaryFn_relu : I32EnumAttrCase<"relu", 0, "relu">;
def Seki_UnaryFn_exp  : I32EnumAttrCase<"exp",  1, "exp">;
def Seki_UnaryFn_log  : I32EnumAttrCase<"log",  2, "log">;
def Seki_UnaryFn_neg  : I32EnumAttrCase<"neg",  3, "neg">;
def Seki_UnaryFn_abs  : I32EnumAttrCase<"abs",  4, "abs">;

def Seki_UnaryFn : I32EnumAttr<"UnaryFn", "unary elementwise function",
    [Seki_UnaryFn_relu, Seki_UnaryFn_exp, Seki_UnaryFn_log,
     Seki_UnaryFn_neg,  Seki_UnaryFn_abs]> {
    let cppNamespace = "::mlir::seki";
    let genSpecializedAttr = 0;
}

def Seki_UnaryFnAttr : EnumAttr<SekiDialect, Seki_UnaryFn, "unary_fn"> {
    let assemblyFormat = "$value";
}

//===-----------------------------------------------===//
// BinaryFn — seki.ew_binary<fn>
//===-----------------------------------------------===//
def Seki_BinaryFn_add : I32EnumAttrCase<"add", 0, "add">;
def Seki_BinaryFn_sub : I32EnumAttrCase<"sub", 1, "sub">;
def Seki_BinaryFn_mul : I32EnumAttrCase<"mul", 2, "mul">;
def Seki_BinaryFn_div : I32EnumAttrCase<"div", 3, "div">;
def Seki_BinaryFn_max : I32EnumAttrCase<"max", 4, "max">;
def Seki_BinaryFn_min : I32EnumAttrCase<"min", 5, "min">;

def Seki_BinaryFn : I32EnumAttr<"BinaryFn", "binary elementwise function",
    [Seki_BinaryFn_add, Seki_BinaryFn_sub, Seki_BinaryFn_mul,
     Seki_BinaryFn_div, Seki_BinaryFn_max, Seki_BinaryFn_min]> {
    let cppNamespace = "::mlir::seki";
    let genSpecializedAttr = 0;
}

def Seki_BinaryFnAttr : EnumAttr<SekiDialect, Seki_BinaryFn, "binary_fn"> {
    let assemblyFormat = "$value";
}

//===-----------------------------------------------===//
// ReduceFn — seki.reduce<fn>
//===-----------------------------------------------===//
def Seki_ReduceFn_sum : I32EnumAttrCase<"sum", 0, "sum">;
def Seki_ReduceFn_max : I32EnumAttrCase<"max", 1, "max">;
def Seki_ReduceFn_min : I32EnumAttrCase<"min", 2, "min">;

def Seki_ReduceFn : I32EnumAttr<"ReduceFn", "reduction function",
    [Seki_ReduceFn_sum, Seki_ReduceFn_max, Seki_ReduceFn_min]> {
    let cppNamespace = "::mlir::seki";
    let genSpecializedAttr = 0;
}

def Seki_ReduceFnAttr : EnumAttr<SekiDialect, Seki_ReduceFn, "reduce_fn"> {
    let assemblyFormat = "$value";
}

//===-----------------------------------------------===//
// Axis — seki.extract<axis> / seki.update<axis>
//===-----------------------------------------------===//
def Seki_Axis_rows : I32EnumAttrCase<"rows", 0, "rows">;
def Seki_Axis_cols : I32EnumAttrCase<"cols", 1, "cols">;

def Seki_Axis : I32EnumAttr<"Axis", "tile axis (row-wise or column-wise)",
    [Seki_Axis_rows, Seki_Axis_cols]> {
    let cppNamespace = "::mlir::seki";
    let genSpecializedAttr = 0;
}

def Seki_AxisAttr : EnumAttr<SekiDialect, Seki_Axis, "axis"> {
    let assemblyFormat = "$value";
}

//===-----------------------------------------------===//
// SekiSupportedOp — target capability list
//===-----------------------------------------------===//
def SekiSupportedOp_Matmul      : I32EnumAttrCase<"Matmul",      0,  "matmul">;
def SekiSupportedOp_BatchMatmul : I32EnumAttrCase<"BatchMatmul", 1,  "batch_matmul">;
def SekiSupportedOp_EwUnaryRelu : I32EnumAttrCase<"EwUnaryRelu", 2,  "ew_unary_relu">;
def SekiSupportedOp_EwUnaryExp  : I32EnumAttrCase<"EwUnaryExp",  3,  "ew_unary_exp">;
def SekiSupportedOp_EwUnaryLog  : I32EnumAttrCase<"EwUnaryLog",  4,  "ew_unary_log">;
def SekiSupportedOp_EwUnaryNeg  : I32EnumAttrCase<"EwUnaryNeg",  5,  "ew_unary_neg">;
def SekiSupportedOp_EwUnaryAbs  : I32EnumAttrCase<"EwUnaryAbs",  6,  "ew_unary_abs">;
def SekiSupportedOp_EwBinaryAdd : I32EnumAttrCase<"EwBinaryAdd", 7,  "ew_binary_add">;
def SekiSupportedOp_EwBinarySub : I32EnumAttrCase<"EwBinarySub", 8,  "ew_binary_sub">;
def SekiSupportedOp_EwBinaryMul : I32EnumAttrCase<"EwBinaryMul", 9,  "ew_binary_mul">;
def SekiSupportedOp_EwBinaryDiv : I32EnumAttrCase<"EwBinaryDiv", 10, "ew_binary_div">;
def SekiSupportedOp_EwBinaryMax : I32EnumAttrCase<"EwBinaryMax", 11, "ew_binary_max">;
def SekiSupportedOp_EwBinaryMin : I32EnumAttrCase<"EwBinaryMin", 12, "ew_binary_min">;
def SekiSupportedOp_ReduceSum   : I32EnumAttrCase<"ReduceSum",   13, "reduce_sum">;
def SekiSupportedOp_ReduceMax   : I32EnumAttrCase<"ReduceMax",   14, "reduce_max">;
def SekiSupportedOp_ReduceMin   : I32EnumAttrCase<"ReduceMin",   15, "reduce_min">;
def SekiSupportedOp_Broadcast   : I32EnumAttrCase<"Broadcast",   16, "broadcast">;
def SekiSupportedOp_Transpose   : I32EnumAttrCase<"Transpose",   17, "transpose">;
def SekiSupportedOp_Reshape     : I32EnumAttrCase<"Reshape",     18, "reshape">;
def SekiSupportedOp_Extract     : I32EnumAttrCase<"Extract",     19, "extract">;
def SekiSupportedOp_Update      : I32EnumAttrCase<"Update",      20, "update">;

def Seki_SupportedOp : I32EnumAttr<"SekiSupportedOp",
    "seki compute op supported by target hardware",
    [SekiSupportedOp_Matmul,   SekiSupportedOp_BatchMatmul,
     SekiSupportedOp_EwUnaryRelu, SekiSupportedOp_EwUnaryExp,
     SekiSupportedOp_EwUnaryLog,  SekiSupportedOp_EwUnaryNeg,
     SekiSupportedOp_EwUnaryAbs,
     SekiSupportedOp_EwBinaryAdd, SekiSupportedOp_EwBinarySub,
     SekiSupportedOp_EwBinaryMul, SekiSupportedOp_EwBinaryDiv,
     SekiSupportedOp_EwBinaryMax, SekiSupportedOp_EwBinaryMin,
     SekiSupportedOp_ReduceSum,   SekiSupportedOp_ReduceMax,
     SekiSupportedOp_ReduceMin,
     SekiSupportedOp_Broadcast, SekiSupportedOp_Transpose,
     SekiSupportedOp_Reshape,
     SekiSupportedOp_Extract, SekiSupportedOp_Update]> {
    let cppNamespace = "::mlir::seki";
    let genSpecializedAttr = 0;
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
// Compute subsystem configuration
//===-----------------------------------------------===//
def SekiComputeConfigAttr : SekiAttr<"SekiComputeConfig", "compute_config"> {
    let summary = "Seki compute subsystem hardware configuration";
    let parameters = (ins
        "int64_t":$macArrayRows,
        "int64_t":$macArrayCols,
        "int64_t":$computeUnits,
        "int64_t":$vectorRegisterFileBytes
    );
    let assemblyFormat = "`<` struct(params) `>`";
    let genVerifyDecl = 1;
}

//===-----------------------------------------------===//
// ISA configuration
//===-----------------------------------------------===//
def SekiISAConfigAttr : SekiAttr<"SekiISAConfig", "isa_config"> {
    let summary = "Seki ISA configuration";
    let parameters = (ins
        "int64_t":$isaVersion,
        "int64_t":$maxTileId,
        ArrayRefParameter<"::mlir::seki::SekiSupportedOp">:$supportedOps
    );
    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

//===-----------------------------------------------===//
// Top-level hardware target
//===-----------------------------------------------===//
def SekiTargetAttr : SekiAttr<"SekiTarget", "target"> {
    let summary = "Seki hardware target description attached to a module";
    let parameters = (ins
        "::mlir::seki::SekiMemoryConfigAttr":$memory,
        "::mlir::seki::SekiComputeConfigAttr":$compute,
        "::mlir::seki::SekiISAConfigAttr":$isa
    );
    let assemblyFormat = "`<` struct(params) `>`";
}


#endif // SEKI_ATTRS_TD
