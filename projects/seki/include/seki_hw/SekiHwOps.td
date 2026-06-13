#ifndef SEKI_HW_OPS_TD
#define SEKI_HW_OPS_TD

include "seki_hw/SekiHwDialect.td"
include "seki_hw/SekiHwTypes.td"
include "seki_hw/SekiHwAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===-----------------------------------------------===//
// SekiHw Op base class
//===-----------------------------------------------===//
class SekiHwOps<string mnemonic, list<Trait> traits = []>
    : Op<SekiHwDialect, mnemonic, traits>;

//===-----------------------------------------------===//
// Ops
//===-----------------------------------------------===//
//===-----------------------------------------------===//
// seki_hw.load - DMA: DRAM -> scratchpad
// %tile = seki_hw.load %addr -> !seki_hw.tile<f32>
//===-----------------------------------------------===//
def SekiHw_LoadOp : SekiHwOps<"load", [MemoryEffects<[MemRead]>]> {
    let summary = "DMA load: DRAM byte address -> scratchpad tile";
    let arguments = (ins I64:$dram_addr);
    let results = (outs SekiHw_TileType:$tile);
    let assemblyFormat = "$dram_addr attr-dict `->` type($tile)";
}

//===-----------------------------------------------===//
// seki_hw.store - DMA: scratchpad tile -> DRAM
// seki_hw.store %tile, %addr -> !seki_hw.tile<f32>
//===-----------------------------------------------===//
def SekiHw_StoreOp : SekiHwOps<"store", [MemoryEffects<[MemWrite]>]> {
    let summary = "DMA load: DRAM byte address -> scratchpad tile";
    let arguments = (ins SekiHw_TileType:$tile, I64:$dram_addr);
    let assemblyFormat = "$tile `,` $dram_addr attr-dict `:` type($tile)";
}

//===-----------------------------------------------===//
// seki_hw.matmul - systolic array matrix multiply-accumulate
// Destination-passing style: $acc is the initial accumulator (C_in)
// $result is the updated accumulator (C_out = C_in + A x B).
// $acc and $result occupy the same physical tile slot after allocation.
// %c_out = seki_hw.matmul ins(%a, %b: tile<Ta>, tile<Tb>)
//                         outs (%c_in : tile<Tc>) -> tile<Tc>
//===-----------------------------------------------===//
def SekiHw_MatmulOp : SekiHwOps<"matmul", [Pure]> {
    let summary = "Systolic array matrix multiply-accumulate: C += A x B";
    let arguments = (ins
        SekiHw_TileType:$lhs,
        SekiHw_TileType:$rhs,
        SekiHw_TileType:$acc
    );
    let results = (outs SekiHw_TileType:$result);
    let assemblyFormat = [{`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
                           `outs` `(` $acc `:` type($acc) `)`
                           attr-dict `->` type($result)}];
    let hasVerifier = 1;
}

//===-----------------------------------------------===//
// seki_hw.tile_ew_unary<fn> - unary elementwise: tile<T> -> tile<T>
// %out = seki_hw.tile_ew_unary<relu> %src : !seki_hw.tile<f32>
//===-----------------------------------------------===//
def SekiHw_TileEwUnaryOp: SekiHwOps<"tile_ew_unary", [Pure, SameOperandsAndResultType]> {
    let summary = "Unary elementwise op on a tile";
    let arguments = (ins SekiHw_UnaryFnAttr:$fn, SekiHw_TileType:$src);
    let results = (outs SekiHw_TileType:$result);
    let assemblyFormat = "`<` $fn `>` $src attr-dict `:`type($result)";
}

//===-----------------------------------------------===//
// seki_hw.tile_ew_binary<fn> - binary elementwise: tile<T> x tile<T> -> tile<T>
// %out = seki_hw.tile_ew_binary<add> %lhs, %rhs : !seki_hw.tile<f32>, !seki_hw.tile<f32>
//===-----------------------------------------------===//
def SekiHw_TileEwBinaryOp : SekiHwOps<"tile_ew_binary", [Pure, SameOperandsAndResultType]> {
    let summary = "Binary elementwise op on tiles";
    let arguments = (ins SekiHw_BinaryFnAttr:$fn, SekiHw_TileType:$lhs, SekiHw_TileType:$rhs);
    let results = (outs SekiHw_TileType:$result);
    let assemblyFormat = "`<` $fn `>` $lhs `,` $rhs attr-dict `:` type($result)";
}

//===-----------------------------------------------===//
// seki_hw.tile_splat - fill tile with scalar: T -> tile<T>
// %tile = seki_hw.tile_splat %scalar : f32 -> !seki_hw.tile<f32>
//===-----------------------------------------------===//
def SekiHw_TileSplatOp : SekiHwOps<"tile_splat", [Pure]> {
    let summary = "Fill a tile with a scalar value";
    let arguments = (ins AnyType:$scalar);
    let results = (outs SekiHw_TileType:$result);
    let assemblyFormat = "$scalar attr-dict `:` type($scalar) `->` type($result)";
    let hasVerifier = 1;
}

//===-----------------------------------------------===//
// seki_hw.tile_reduce_vec<fn, axis> - partial tile reduction: tile<T> -> vec<T, N>
// %tile = seki_hw.tile_reduce_vec<sum, rows> %src : !seki_hw.vec<f32> -> !seki_hw.vec<f32, N>
//===-----------------------------------------------===//
def SekiHw_TileReduceVecOp : SekiHwOps<"tile_reduce_vec", [Pure]> {
    let summary = "Partial tile reduction along rows or cols: tile<T> -> vec<T, N>";
    let arguments = (ins SekiHw_ReduceFnAttr:$fn, SekiHw_TileAxisAttr: $axis, SekiHw_TileType:$src);
    let results = (outs SekiHw_VecType:$result);
    let assemblyFormat = "`<` $fn `,` $axis `>` $src attr-dict `:` type($src) `->` type($result)";
    let hasVerifier = 1;
}

//===-----------------------------------------------===//
// seki_hw.vec_broadcast_tile - broadcast vec to tile: vec<T, N> -> tile<T>
// %tile = seki_hw.vec_broadcast_tile<rows> %vec : !seki_hw.vec<f32, 128> -> !seki_hw.tile<f32>
//===-----------------------------------------------===//
def SekiHw_VecBroadcastTileOp : SekiHwOps<"vec_broadcast_tile", [Pure]> {
    let summary = "Broadcast a vec to fill a tile along rows or cols";
    let arguments = (ins SekiHw_TileAxisAttr: $axis, SekiHw_VecType:$vec);
    let results = (outs SekiHw_TileType:$result);
    let assemblyFormat = "`<` $axis `>` $vec attr-dict `:` type($vec) `->` type($result)";
    let hasVerifier = 1;
}

#endif // SEKI_HW_OPS_TD
