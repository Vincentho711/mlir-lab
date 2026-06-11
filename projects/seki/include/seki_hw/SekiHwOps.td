#ifndef SEKI_HW_OPS_TD
#define SEKI_HW_OPS_TD

include "seki_hw/SekiHwDialect.td"
include "seki_hw/SekiHwTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

#endif // SEKI_HW_OPS_TD
