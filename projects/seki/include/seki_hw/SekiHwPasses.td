#ifndef SEKI_HW_PASSES_TD
#define SEKI_HW_PASSES_TD
include "mlir/Pass/PassBase.td"

def DumpTileEwOpsPass : Pass<"seki-hw-dump-tile-ew-ops", "::mlir::func::FuncOp"> {
    let summary = "Diagnostic: emit a remark for every tile_ew_unary/binary showing getEwFn()";
    let description = [{
        Walks the function body, and for every op implementing SekiTileEwOpInterface, emits a remark
        with the op name and the attribute returned by getEwFn(). Used to verify that SekiTileEwOpInterface dispatch
        is wired up correctly for both TileEwUnaryOp and TileEwBinaryOp.
    }];
    let dependentDialects = ["::mlir::seki_hw::SekiHwDialect"];
}

#endif // SEKI_HW_PASSES_TD
