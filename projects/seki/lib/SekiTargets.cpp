#include "seki/SekiTargets.h"
#include "seki/SekiAttrs.h"

namespace mlir::seki {

static SekiTargetAttr buildSekiV1(MLIRContext *ctx) {
    auto mem = SekiMemoryConfigAttr::get(ctx,
        /*scratchpadBytes=*/     262144,
        /*dramBytes=*/           4294967296LL,
        /*dmaAlignment=*/        64,
        /*numScratchpadBanks=*/  4);
    return SekiTargetAttr::get(ctx, mem);
}

SekiTargetAttr getBuiltinTarget(llvm::StringRef name, MLIRContext *ctx) {
    if (name == "seki-v1") return buildSekiV1(ctx);
    return nullptr;
}
} // namespace mlir::seki
