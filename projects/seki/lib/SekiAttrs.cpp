#include "seki/SekiAttrs.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"

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

} // namespace mlir::seki
