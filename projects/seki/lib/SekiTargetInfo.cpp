#include "seki/SekiTargetInfo.h"
#include "seki/SekiAttrs.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::seki {
SekiTargetInfo::SekiTargetInfo(mlir::ModuleOp mod) {
    auto target = mod->getAttrOfType<SekiTargetAttr>("seki.target");
    if (!target)
        llvm::report_fatal_error("module has no #seki.target attribute - "
                                 "pass --seki-target to seki-opt");

    SekiMemoryConfigAttr mem = target.getMemory();
    scratchpadBytes    = mem.getScratchpadBytes();
    dramBytes          = mem.getDramBytes();
    dmaAlignment       = mem.getDmaAlignment();
    numScratchpadBanks = mem.getNumScratchpadBanks();

    SekiComputeConfigAttr compute = target.getCompute();
    macArrayRows            = compute.getMacArrayRows();
    macArrayCols            = compute.getMacArrayCols();
    computeUnits            = compute.getComputeUnits();
    vectorRegisterFileBytes = compute.getVectorRegisterFileBytes();

    SekiISAConfigAttr isa = target.getIsa();
    isaVersion = isa.getIsaVersion();
    maxTileId  = isa.getMaxTileId();
}

int64_t SekiTargetInfo::getScratchpadBytes()           const { return scratchpadBytes; }
int64_t SekiTargetInfo::getDRAMBytes()                 const { return dramBytes; }
int64_t SekiTargetInfo::getDMAAlignment()              const { return dmaAlignment; }
int64_t SekiTargetInfo::getNumScratchpadBanks()        const { return numScratchpadBanks; }
int64_t SekiTargetInfo::getMACRows()                   const { return macArrayRows; }
int64_t SekiTargetInfo::getMACCols()                   const { return macArrayCols; }
int64_t SekiTargetInfo::getComputeUnits()              const { return computeUnits; }
int64_t SekiTargetInfo::getVectorRegisterFileBytes()   const { return vectorRegisterFileBytes; }
int64_t SekiTargetInfo::getVecRegisterSizeBytes(mlir::Type elementType) const {
    return macArrayCols * (elementType.getIntOrFloatBitWidth() / 8);
}
int64_t SekiTargetInfo::getVecRegisterCount(mlir::Type elementType) const {
    return vectorRegisterFileBytes / getVecRegisterSizeBytes(elementType);
}
int64_t SekiTargetInfo::getISAVersion()                const { return isaVersion; }
int64_t SekiTargetInfo::getMaxTileId()                 const { return maxTileId; }
int64_t SekiTargetInfo::getMaxTileSlots()              const { return maxTileId + 1; }

} // namespace mlir::seki
