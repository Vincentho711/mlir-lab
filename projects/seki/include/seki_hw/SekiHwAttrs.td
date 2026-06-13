#ifndef SEKI_HW_ATTRS_TD
#define SEKI_HW_ATTRS_TD

include "seki_hw/SekiHwDialect.td"
include "mlir/IR/EnumAttr.td"

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


#endif // SEKI_HW_ATTRS_TD
