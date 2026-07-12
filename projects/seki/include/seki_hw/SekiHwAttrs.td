#ifndef SEKI_HW_ATTRS_TD
#define SEKI_HW_ATTRS_TD

include "seki_hw/SekiHwDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

//===-------------------------------------------------------------===//
// UnaryFn - single-input elementwise : tile_ew_unary, vec_ew_unary
//===-------------------------------------------------------------===//
def SekiHw_UnaryFn_relu : I32EnumAttrCase<"relu", 0, "relu">; // <name, int, printed keyword>
def SekiHw_UnaryFn_exp : I32EnumAttrCase<"exp", 1, "exp">;
def SekiHw_UnaryFn_log : I32EnumAttrCase<"log", 2, "log">;
def SekiHw_UnaryFn_neg : I32EnumAttrCase<"neg", 3, "neg">;
def SekiHw_UnaryFn_abs : I32EnumAttrCase<"abs", 4, "abs">;

// Group cases into a C++ enum class
// Generating enum class mlir::seki_hw::UnaryFn {relu = 0, ...}.
def SekiHw_UnaryFn : I32EnumAttr<"UnaryFn", "unary elementwise function",
    [SekiHw_UnaryFn_relu, SekiHw_UnaryFn_exp, SekiHw_UnaryFn_log, SekiHw_UnaryFn_neg, SekiHw_UnaryFn_abs]> {
    let cppNamespace = "::mlir::seki_hw";
    let genSpecializedAttr = 0; // Don't generate an MLIR attribute wrapper here.
}

def SekiHw_UnaryFnAttr : EnumAttr<SekiHwDialect, SekiHw_UnaryFn, "unary_fn"> {
    let assemblyFormat = "$value";
}

//===-------------------------------------------------------------===//
// BinaryFn - two-input elementwise : tile_ew_binary, vec_ew_binary
//===-------------------------------------------------------------===//
def SekiHw_BinaryFn_add : I32EnumAttrCase<"add", 0, "add">;
def SekiHw_BinaryFn_sub : I32EnumAttrCase<"sub", 1, "sub">;
def SekiHw_BinaryFn_mul : I32EnumAttrCase<"mul", 2, "mul">;
def SekiHw_BinaryFn_div : I32EnumAttrCase<"div", 3, "div">;
def SekiHw_BinaryFn_max : I32EnumAttrCase<"max", 4, "max">;
def SekiHw_BinaryFn_min : I32EnumAttrCase<"min", 5, "min">;

def SekiHw_BinaryFn : I32EnumAttr<"BinaryFn", "binary elementwise function",
    [SekiHw_BinaryFn_add, SekiHw_BinaryFn_sub, SekiHw_BinaryFn_mul, SekiHw_BinaryFn_div, SekiHw_BinaryFn_max, SekiHw_BinaryFn_min]> {
    let cppNamespace = "::mlir::seki_hw";
    let genSpecializedAttr = 0;
}

def SekiHw_BinaryFnAttr : EnumAttr<SekiHwDialect, SekiHw_BinaryFn, "binary_fn"> {
    let assemblyFormat = "$value";
}

//===-------------------------------------------------------------===//
// ReduceFn - N-ary reduction : tile_reduce_vec, tile_reduce_scalar
//===-------------------------------------------------------------===//
def SekiHw_ReduceFn_sum : I32EnumAttrCase<"sum", 0, "sum">;
def SekiHw_ReduceFn_max : I32EnumAttrCase<"max", 1, "max">;
def SekiHw_ReduceFn_min : I32EnumAttrCase<"min", 2, "min">;

def SekiHw_ReduceFn : I32EnumAttr<"ReduceFn", "reduction function",
    [SekiHw_ReduceFn_sum, SekiHw_ReduceFn_max, SekiHw_ReduceFn_min]> {
    let cppNamespace = "::mlir::seki_hw";
    let genSpecializedAttr = 0;
}

// Produces a C++ class mlir::seki_hw::ReduceFnAttr with:
// static ReduceFnAttr get(MLIRContext *ctx, ReduceFn value);
def SekiHw_ReduceFnAttr : EnumAttr<SekiHwDialect, SekiHw_ReduceFn, "reduce_fn"> {
    let assemblyFormat = "$value";
}

//===-------------------------------------------------------------===//
// TileAxis - axis for tile_reduce_vec / vec_broadcast_tile
//===-------------------------------------------------------------===//
def SekiHw_TileAxis_rows : I32EnumAttrCase<"rows", 0, "rows">;
def SekiHw_TileAxis_cols : I32EnumAttrCase<"cols", 1, "cols">;

def SekiHw_TileAxis : I32EnumAttr<"TileAxis", "tile axis",
    [SekiHw_TileAxis_rows, SekiHw_TileAxis_cols]> {
    let cppNamespace = "::mlir::seki_hw";
    let genSpecializedAttr = 0; // only generate enum class ReduceAxis {rows=0, cols=1}. Don't generate an MLIR attribute class.
    // We will implement our own SekiHw_TileAxisAttr with our own mnemonic and assemblyFormat for shorter inline syntax and clean qualified name
}

def SekiHw_TileAxisAttr: EnumAttr<SekiHwDialect, SekiHw_TileAxis, "tile_axis"> {
    let assemblyFormat = "$value";
}

//===-------------------------------------------------------------===//
// TernaryFn - three-input elementwise : tile_ew_ternary
//===-------------------------------------------------------------===//
def SekiHw_TernaryFn_fma : I32EnumAttrCase<"fma", 0, "fma">;

def SekiHw_TernaryFn : I32EnumAttr<"TernaryFn", "ternary elementwise function",
    [SekiHw_TernaryFn_fma]> {
    let cppNamespace = "::mlir::seki_hw";
    let genSpecializedAttr = 0;
}

def SekiHw_TernaryFnAttr : EnumAttr<SekiHwDialect, SekiHw_TernaryFn, "ternary_fn"> {
    let assemblyFormat = "$value";
}


//===-------------------------------------------------------------===//
// HardwareUnit - execution unit vocabulary shared between ops and target
//===-------------------------------------------------------------===//
def SekiHw_HardwareUnit_MACArray : I32EnumAttrCase<"MACArray", 0, "mac_array">;
def SekiHw_HardwareUnit_VecEngine : I32EnumAttrCase<"VecEngine", 1, "vec_engine">;
def SekiHw_HardwareUnit_ScalarUnit : I32EnumAttrCase<"ScalarUnit", 2, "scalar_unit">;
def SekiHw_HardwareUnit_DMAEngine : I32EnumAttrCase<"DMAEngine", 3, "dma_engine">;

def SekiHw_HardwareUnit : I32EnumAttr<"HardwareUnit", "seki hardware execution unit",
    [SekiHw_HardwareUnit_MACArray, SekiHw_HardwareUnit_VecEngine,
     SekiHw_HardwareUnit_ScalarUnit, SekiHw_HardwareUnit_DMAEngine]> {
    let cppNamespace = "::mlir::seki_hw";
    let genSpecializedAttr = 0;
}

//===-------------------------------------------------------------===//
// PerUnit capability attrs
// Each attr describes what one hardware unit can do.
//===-------------------------------------------------------------===//

// MAC array: what element types A/B must be, and what output types the accumulator can
// write to scratchpad.
// inputType:   single type - A and B tiles must both have this element type
// outputTypes:  array of types - the IR outs type select which output mode fires
//               (f32 = preserve accumulator precision across K-tiles; bf16 truncate on write)
def SekiHw_MACArrayCapAttr : SekiHwAttr<"MACArrayCap", "mac_array"> {
    let summary = "MAC array capability: input element type and supported output types";
    let parameters = (ins "::mlir::Type":$inputType, "::mlir::ArrayAttr":$outputTypes);
    // assemblyFormat doesn't work when parameters are ::mlir::Type or ::mlir::ArrayAttr as TableGen doesn't know how to parse and print them
    let hasCustomAssemblyFormat = 1;
    // declare a static verify(emitError, params, ...) method in generated header.
    // Used to ensure that both inputTypes and outputTypes must be float types
    let genVerifyDecl = 1;
}

// VEC engine: element types for homogeneous ops (relu, add - same in/out type), and (from, to) pairs
// for tile_cast (type-converting ops).
// computeTypes: array of types - vec can execute ew ops on these
// castPairs: array of (Type, Type) pairs - vec can convert between these
def SekiHw_VecEngineCapAttr : SekiHwAttr<"VecEngineCap", "vec_engine"> {
    let summary = "VEC engine capability: compute types and cast pairs";
    let parameters = (ins "::mlir::ArrayAttr":$computeTypes, "::mlir::ArrayAttr":$castPairs);
    let hasCustomAssemblyFormat = 1;
}

// Scalar unit: integer types it can compute on.
def SekiHw_ScalarUnitCapAttr : SekiHwAttr<"ScalarUnitCap", "scalar_unit"> {
    let summary = "Scalar unit capability: supported integer types";
    let parameters = (ins "::mlir::ArrayAttr":$supportedTypes);
    let hasCustomAssemblyFormat = 1;
    // Verify to ensure that supportedTypes must be IntegerTypes
    let genVerifyDecl = 1;
}

// DMA engine: (from, to) type pairs it can convert inline during a transfer (load or store), separated by direction.
// loadCastPairs: pairs legal for dma_load_cast (DRAM -> tiles)
// storeCastPairs: pairs legal for dma_store_cast (tiles -> DRAM)
// Empty list = no cast support in that direction
// No verifer: empty lists are valid (targets without DMA cast support).
def SekiHw_DMAEngineCapAttr : SekiHwAttr<"DMAEngineCap", "dma_engine"> {
    let summary = "DMA engine capability: inline cast pairs per transfer direction";
    let parameters = (ins "::mlir::ArrayAttr":$loadCastPairs, "::mlir::ArrayAttr":$storeCastPairs);
    let hasCustomAssemblyFormat = 1;
}

//===-------------------------------------------------------------===//
// ISAConfigAttr - store all the unit capabilities for a seki_hw target
//===-------------------------------------------------------------===//
def SekiHw_ISAConfigAttr : SekiHwAttr<"ISAConfig", "isa_config"> {
    let summary = "ISA configuration: per-unit capabilities for a seki_hw target";
    let parameters = (ins
        "::mlir::Attribute":$macArray,
        "::mlir::Attribute":$vecEngine,
        "::mlir::Attribute":$scalarUnit,
        "::mlir::Attribute":$dmaEngine
    );
    let hasCustomAssemblyFormat = 1;
    // As parameters are typed ::mlir::Attribute, we need to ensure that macArray is actually a
    // MACArrayCapAttr using verify()
    let genVerifyDecl = 1;
}

#endif // SEKI_HW_ATTRS_TD
