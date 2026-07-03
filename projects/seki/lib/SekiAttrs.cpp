#include "seki/SekiAttrs.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallVector.h"

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
    int64_t maxTileId,
    llvm::ArrayRef<SekiSupportedOp> supportedOps)
{
    if (isaVersion <= 0)
        return emitError() << "isa_version must be positive";
    if (maxTileId <= 0)
        return emitError() << "max_tile_id must be positive";
    return mlir::success();
}

mlir::Attribute SekiISAConfigAttr::parse(::mlir::AsmParser &parser, ::mlir::Type) {
    // Save location before consuming any tokens so verify() errors point to the
    // attribute in the source file rather than <unknown>:0 (which would happen
    // with get(), which asserts instead of emitting a diagnosable error).
    auto attrLoc = parser.getCurrentLocation();
    if (parser.parseLess()) return {};
    int64_t isaVersion, maxTileId;
    if (parser.parseKeyword("isaVersion") || parser.parseEqual() ||
        parser.parseInteger(isaVersion)   || parser.parseComma() ||
        parser.parseKeyword("maxTileId")  || parser.parseEqual() ||
        parser.parseInteger(maxTileId)    || parser.parseComma() ||
        parser.parseKeyword("supported_ops") || parser.parseEqual())
        return {};

    llvm::SmallVector<SekiSupportedOp> ops;
    auto parseOne = [&]() -> ::mlir::ParseResult {
        llvm::StringRef kw;
        auto loc = parser.getCurrentLocation();
        if (parser.parseKeyword(&kw)) return ::mlir::failure();
        // symbolizeSekiSupportedOp(kw) returns std::optional<SekiSupportedOp>
        auto sym = symbolizeSekiSupportedOp(kw);
        if (!sym)
            return parser.emitError(loc) << "unknown SekiSupportedOp: " << kw;
        // if it exists as an op, add it to the supported vector
        ops.push_back(*sym);
        return ::mlir::success();
    };

    // parseCommaSeparatedList(Delimiter::Square, parseOne) handles the full [a, b, c] bracket syntax, including empty list.
    if (parser.parseCommaSeparatedList(::mlir::AsmParser::Delimiter::Square, parseOne))
        return {};
    if (parser.parseGreater()) return {};

    // Verify before get() so invalid values emit a parse-location diagnostic
    // rather than triggering an assertion (getChecked can't be used here because
    // it's a template that requires the complete storage type).
    auto emitErr = [&]() { return parser.emitError(attrLoc); };
    if (failed(SekiISAConfigAttr::verify(emitErr, isaVersion, maxTileId, ops)))
        return {};
    return SekiISAConfigAttr::get(parser.getContext(), isaVersion, maxTileId, ops);
}

void SekiISAConfigAttr::print(::mlir::AsmPrinter &printer) const {
    printer << "<isaVersion = " << getIsaVersion()
            << ", maxTileId = " << getMaxTileId()
            << ", supported_ops = [";
    llvm::interleaveComma(getSupportedOps(), printer.getStream(), [&](SekiSupportedOp op) {
        printer.getStream() << stringifySekiSupportedOp(op);
    });
    printer << "]>";
}

} // namespace mlir::seki
