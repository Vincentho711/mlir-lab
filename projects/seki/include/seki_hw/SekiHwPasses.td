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

def SekiHwVerifyTargetPass : Pass<"seki-hw-verify-target", "::mlir::ModuleOp"> {
    let summary = "Verify IR is physically realisable on the attached SekiTargetAttr";
    let description = [{
        Reads the #seki.target attribute attached to the module and checks:
        1. ISA version matches the compiler's expected version (hard error).
        2. Peak simulaneously-live vec<T> values do not exceed vec register file
           capacity per element type (hard error).
        3. Peak simultaneously-live tile<T> values do not exceed the hardware slot
           count (advisory warning; hard enforcement deferred).
        Checks 2 and 3 use MLIR liveness analysis per function.
    }];
    let dependentDialects = [
        "::mlir::seki_hw::SekiHwDialect",
        "::mlir::func::FuncDialect"
    ];
}

#endif // SEKI_HW_PASSES_TD
