#ifndef SEKI_HW_OPTS_TD
#define SEKI_HW_OPTS_TD
include "seki_hw/SekiHwDialect.td"

//===-----------------------------------------------===//
// TileType - !seki_hw.tile<T>
// A 64 KB on-chip scratchpad slot. T ∈ {f32, bf16, i8, i32}.
//===-----------------------------------------------===//
def SekiHw_TileType : SekiHwType<"Tile", "tile"> {
    let summary = "64 KB scratchpad slot typed by element type";
    let parameters = (ins "::mlir::Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
    let genVerifyDecl = 1;
}


//===-----------------------------------------------===//
// VecType - !seki_hw.vec<T>
// 2048-byte vector register file value. N = 128 for seki-v1.
//===-----------------------------------------------===//
def SekiHw_VecType : SekiHwType<"Vec", "vec"> {
    let summary = "512-byte vector register file value";
    let parameters = (ins "::mlir::Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
    let genVerifyDecl = 1;
}

#endif // SEKI_HW_OPTS_TD
