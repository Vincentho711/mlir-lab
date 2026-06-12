#include "seki/SekiTargets.h"
#include "seki/SekiAttrs.h"

namespace mlir::seki {

static SekiTargetAttr buildSekiV1(MLIRContext *ctx) {
    auto mem = SekiMemoryConfigAttr::get(ctx,
        /*scratchpadBytes=*/    8388608,
        /*dramBytes=*/          17179869184LL,
        /*dmaAlignment=*/       64,
        /*numScratchpadBanks=*/ 4);
    auto compute = SekiComputeConfigAttr::get(ctx,
        /*macArrayRows=*/             128,
        /*macArrayCols=*/             128,
        /*computeUnits=*/             4,
        /*vectorRegisterFileBytes =*/ 2048);
    auto isa = SekiISAConfigAttr::get(ctx,
        /*isaVersion=*/ 1,
        /*maxTileId=*/  64);
    return SekiTargetAttr::get(ctx, mem, compute, isa);
}

SekiTargetAttr getBuiltinTarget(llvm::StringRef name, MLIRContext *ctx) {
    if (name == "seki-v1") return buildSekiV1(ctx);
    return nullptr;
}
} // namespace mlir::seki
