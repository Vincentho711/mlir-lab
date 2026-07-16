#include "seki/SekiTargetInfo.h"
#include "seki/SekiAttrs.h"
#include "mlir/IR/BuiltinOps.h"

// Helper function to extract SmallVector<type> from ArrayAttr where each element is a TypeAttr
static llvm::SmallVector<mlir::Type> typesFromArrayAttr(mlir::ArrayAttr arr) {
    llvm::SmallVector<mlir::Type> result;
    for (mlir::Attribute a : arr)
        result.push_back(mlir::cast<mlir::TypeAttr>(a).getValue());
    return result;
}

// Helper function to extract SmallVector<pair<Type, Type>> from ArrayAttr where each element is a 2-element ArrayAttr of TypeAttr
static llvm::SmallVector<std::pair<mlir::Type, mlir::Type>> pairsFromArrayAttr(mlir::ArrayAttr arr) {
    llvm::SmallVector<std::pair<mlir::Type, mlir::Type>> result;
    for (mlir::Attribute a : arr) {
        auto pair = mlir::cast<mlir::ArrayAttr>(a);
        result.push_back({
            mlir::cast<mlir::TypeAttr>(pair[0]).getValue(),
            mlir::cast<mlir::TypeAttr>(pair[1]).getValue()
        });
    }
    return result;
}

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
    maxTileId          = mem.getMaxTileId();

    SekiComputeConfigAttr compute = target.getCompute();
    macArrayRows            = compute.getMacArrayRows();
    macArrayCols            = compute.getMacArrayCols();
    computeUnits            = compute.getComputeUnits();
    vectorRegisterFileBytes = compute.getVectorRegisterFileBytes();

    isaVersion = target.getIsaVersion();

    seki_hw::ISAConfigAttr isa = target.getHwConfig();
    if (auto a = isa.getMacArray())
        macArrayCap = mlir::cast<seki_hw::MACArrayCapAttr>(a);
    if (auto a = isa.getVecEngine())
        vecEngineCap = mlir::cast<seki_hw::VecEngineCapAttr>(a);
    if (auto a = isa.getScalarUnit())
        scalarUnitCap = mlir::cast<seki_hw::ScalarUnitCapAttr>(a);
    if (auto a = isa.getDmaEngine())
        dmaEngineCap = mlir::cast<seki_hw::DMAEngineCapAttr>(a);
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

bool SekiTargetInfo::hasMACArray()                     const { return static_cast<bool>(macArrayCap); }
bool SekiTargetInfo::hasVecEngine()                    const { return static_cast<bool>(vecEngineCap); }
bool SekiTargetInfo::hasUnit(mlir::seki_hw::HardwareUnit unit) const {
    switch (unit) {
        case seki_hw::HardwareUnit::MACArray:  return hasMACArray();
        case seki_hw::HardwareUnit::VecEngine: return hasVecEngine();
        case seki_hw::HardwareUnit::ScalarUnit: return static_cast<bool>(scalarUnitCap);
        case seki_hw::HardwareUnit::DMAEngine: return static_cast<bool>(dmaEngineCap);
    }
    llvm_unreachable("unknown HardwareUnit");
}

// Per-unit query methods
mlir::Type SekiTargetInfo::macInputType() const {
    return macArrayCap ? macArrayCap.getInputType() : mlir::Type{};
}
llvm::SmallVector<mlir::Type> SekiTargetInfo::macOutputTypes() const {
    return macArrayCap ? typesFromArrayAttr(macArrayCap.getOutputTypes()) : llvm::SmallVector<mlir::Type>{};
}
llvm::SmallVector<mlir::Type> SekiTargetInfo::vecComputeTypes() const {
    return vecEngineCap ? typesFromArrayAttr(vecEngineCap.getComputeTypes()) : llvm::SmallVector<mlir::Type>{};
}
llvm::SmallVector<std::pair<mlir::Type,mlir::Type>> SekiTargetInfo::vecCastPairs() const {
    return vecEngineCap ? pairsFromArrayAttr(vecEngineCap.getCastPairs()) : decltype(vecCastPairs()){};
}
llvm::SmallVector<mlir::Type> SekiTargetInfo::scalarSupportedTypes() const {
    return scalarUnitCap ? typesFromArrayAttr(scalarUnitCap.getSupportedTypes()) : llvm::SmallVector<mlir::Type>{};
}
llvm::SmallVector<std::pair<mlir::Type,mlir::Type>> SekiTargetInfo::dmaLoadCastPairs() const {
    return dmaEngineCap ? pairsFromArrayAttr(dmaEngineCap.getLoadCastPairs()) : decltype(dmaLoadCastPairs()){};
}
llvm::SmallVector<std::pair<mlir::Type,mlir::Type>> SekiTargetInfo::dmaStoreCastPairs() const {
    return dmaEngineCap ? pairsFromArrayAttr(dmaEngineCap.getStoreCastPairs()) : decltype(dmaStoreCastPairs()){};
}

// Checks whether the specified unit can execute an op with the given element types
bool SekiTargetInfo::isTypeLegalForUnit(mlir::seki_hw::HardwareUnit unit, llvm::ArrayRef<mlir::Type> inputTypes,
                                        llvm::ArrayRef<mlir::Type> outputTypes) const {
    switch(unit) {
    case seki_hw::HardwareUnit::MACArray: {
        mlir::Type expected = macInputType();
        for (mlir::Type t : inputTypes)
            if (t != expected) return false;
        auto outs = macOutputTypes();
        for (mlir::Type t : outputTypes)
            if (llvm::find(outs, t) == outs.end()) return false;
        return true;
    }
    case seki_hw::HardwareUnit::VecEngine: {
        auto supported = vecComputeTypes();
        for (mlir::Type t : inputTypes)
            if (llvm::find(supported, t) == supported.end()) return false;
        for (mlir::Type t : outputTypes)
            if (llvm::find(supported, t) == supported.end()) return false;
        return true;
    }
    case seki_hw::HardwareUnit::ScalarUnit: {
        auto supported = scalarSupportedTypes();
        for (mlir::Type t : inputTypes)
            if (llvm::find(supported, t) == supported.end()) return false;
        for (mlir::Type t : outputTypes)
            if (llvm::find(supported, t) == supported.end()) return false;
        return true;
    }
    case seki_hw::HardwareUnit::DMAEngine: {
        return hasUnit(unit); // DMA type legality is cast-pair specific; use isCastSupported()
    }
    llvm_unreachable("unknown HardwareUnit");
    }
}

// Checks if casting is supported for all 3 hardware paths
bool SekiTargetInfo::isCastSupported(mlir::Type from, mlir::Type to) const {
    auto inPair = [&](llvm::SmallVector<std::pair<mlir::Type,mlir::Type>> pairs) {
        return llvm::any_of(pairs, [&](auto p) { return p.first == from && p.second == to; });
    };
    return inPair(vecCastPairs()) || inPair(dmaLoadCastPairs()) || inPair(dmaStoreCastPairs());
}


} // namespace mlir::seki
