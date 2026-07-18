#include "seki/SekiTargets.h"
#include "seki/SekiAttrs.h"
#include "seki_hw/SekiHwAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::seki {

static SekiTargetAttr buildSekiV1(MLIRContext *ctx) {
    auto mem = SekiMemoryConfigAttr::get(ctx,
        /*scratchpadBytes=*/    8388608,
        /*dramBytes=*/          17179869184LL,
        /*dmaAlignment=*/       64,
        /*numScratchpadBanks=*/ 4,
        /*maxTileId=*/          63);
    auto compute = SekiComputeConfigAttr::get(ctx,
        /*macArrayRows=*/             128,
        /*macArrayCols=*/             128,
        /*computeUnits=*/             4,
        /*vectorRegisterFileBytes =*/ 2048);

     // MAC array: bf16 input, f32 or bf16 output
    auto macArray = seki_hw::MACArrayCapAttr::get(ctx,
        mlir::BFloat16Type::get(ctx),
        mlir::ArrayAttr::get(ctx, {
            mlir::TypeAttr::get(mlir::Float32Type::get(ctx)),
            mlir::TypeAttr::get(mlir::BFloat16Type::get(ctx))}));
    // VEC engine: bf16 and f32 compute; tile_cast supported in both directions
    auto bf16 = mlir::TypeAttr::get(mlir::BFloat16Type::get(ctx));
    auto f32  = mlir::TypeAttr::get(mlir::Float32Type::get(ctx));
    auto pair = [&](mlir::TypeAttr a, mlir::TypeAttr b) {
        return mlir::ArrayAttr::get(ctx, {a, b});
    };
    auto vecEngine = seki_hw::VecEngineCapAttr::get(ctx,
        mlir::ArrayAttr::get(ctx, {bf16, f32}),
        mlir::ArrayAttr::get(ctx, {pair(f32, bf16), pair(bf16, f32)}));
    // Scalar unit: i64
    auto scalarUnit = seki_hw::ScalarUnitCapAttr::get(ctx,
        mlir::ArrayAttr::get(ctx, {
            mlir::TypeAttr::get(mlir::IntegerType::get(ctx, 64))}));
    // DMA engine: no inline cast support
    auto dmaEngine = seki_hw::DMAEngineCapAttr::get(ctx,
        mlir::ArrayAttr::get(ctx, {}),
        mlir::ArrayAttr::get(ctx, {}));

    auto hwConfig = seki_hw::ISAConfigAttr::get(ctx,
        macArray, vecEngine, scalarUnit, dmaEngine);

    return SekiTargetAttr::get(ctx, mem, compute, /*isaVersion*/1, hwConfig);
}

SekiTargetAttr getBuiltinTarget(llvm::StringRef name, MLIRContext *ctx) {
    if (name == "seki-v1") return buildSekiV1(ctx);
    return nullptr;
}
} // namespace mlir::seki
