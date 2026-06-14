#include "seki/SekiAttrs.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"

#include "seki/SekiEnums.cpp.inc"

namespace mlir::seki {

mlir::LogicalResult SekiMemoryConfigAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t scratchpadBytes,
    int64_t dramBytes,
    int64_t dmaAlignment,
    int64_t numScratchpadBanks)
{
    if (scratchpadBytes <= 0 || (scratchpadBytes & (scratchpadBytes - 1)) != 0)
        return emitError() << "scratchpad_bytes must be a positive power of 2";
    if (dramBytes <= 0 || (dramBytes & (dramBytes - 1)) != 0)
        return emitError() << "dram_bytes must be a positive power of 2";
    if (dmaAlignment <= 0 || (dmaAlignment & (dmaAlignment - 1)) != 0)
        return emitError() << "dma_alignment must be a positive power of 2";
    if (numScratchpadBanks <= 0 || (numScratchpadBanks & (numScratchpadBanks - 1)) != 0)
        return emitError() << "num_scratchpad_banks must be a positive power of 2";
    if (scratchpadBytes % dmaAlignment != 0)
        return emitError() << "scratchpad_bytes (" << scratchpadBytes
                           << ") must be a multiple of dma_alignment ("
                           << dmaAlignment << ")";
    return mlir::success();
}

mlir::LogicalResult SekiComputeConfigAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t macArrayRows,
    int64_t macArrayCols,
    int64_t computeUnits,
    int64_t vectorRegisterFileBytes)
{
    auto isPow2 = [](int64_t v) { return v > 0 && (v & (v - 1)) == 0; };
    if (!isPow2(macArrayRows))
        return emitError() << "mac_array_rows must be a positive power of 2";
    if (!isPow2(macArrayCols))
        return emitError() << "mac_array_cols must be a positive power of 2";
    if (computeUnits <= 0)
        return emitError() << "compute_units must be positive";
    if (!isPow2(vectorRegisterFileBytes))
        return emitError() << "vector_register_file_bytes must be a positive power of 2";
    if (vectorRegisterFileBytes % macArrayCols != 0)
        return emitError() << "vector_register_file_bytes (" << vectorRegisterFileBytes
                           << ") must be divisible by mac_array_cols (" << macArrayCols << ")";
    return mlir::success();
}

mlir::LogicalResult SekiISAConfigAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t isaVersion,
    int64_t maxTileId)
{
    if (isaVersion <= 0)
        return emitError() << "isa_version must be positive";
    if (maxTileId <= 0)
        return emitError() << "max_tile_id must be positive";
    return mlir::success();
}

} // namespace mlir::seki
