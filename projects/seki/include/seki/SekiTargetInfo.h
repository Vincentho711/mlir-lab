#ifndef SEKI_TARGET_INFO_H
#define SEKI_TARGET_INFO_H

#include "mlir/IR/BuiltinOps.h"
#include "seki/SekiAttrs.h"
#include "seki_hw/SekiHwAttrs.h"

namespace mlir::seki {

class SekiTargetInfo {
public:
    explicit SekiTargetInfo(mlir::ModuleOp mod);

    // Memory
    int64_t getScratchpadBytes() const;
    int64_t getDRAMBytes() const;
    int64_t getDMAAlignment() const;
    int64_t getNumScratchpadBanks() const;

    // Compute
    int64_t getMACRows() const;
    int64_t getMACCols() const;
    int64_t getComputeUnits() const;
    int64_t getVectorRegisterFileBytes() const;
    int64_t getVecRegisterSizeBytes(mlir::Type elementType) const;
    int64_t getVecRegisterCount(mlir::Type elementType) const;

    // ISA
    int64_t getISAVersion() const;
    int64_t getMaxTileId() const;
    int64_t getMaxTileSlots() const;

    // Unit presence
    bool hasUnit(mlir::seki_hw::HardwareUnit unit) const;
    bool hasMACArray() const;
    bool hasVecEngine() const;

    // Per-unit capability queries
    mlir::Type macInputType() const;
    llvm::SmallVector<mlir::Type> macOutputTypes() const;
    llvm::SmallVector<mlir::Type> vecComputeTypes() const;
    llvm::SmallVector<std::pair<mlir::Type, mlir::Type>> vecCastPairs() const;
    llvm::SmallVector<mlir::Type> scalarSupportedTypes() const;
    llvm::SmallVector<std::pair<mlir::Type, mlir::Type>> dmaLoadCastPairs() const;
    llvm::SmallVector<std::pair<mlir::Type, mlir::Type>> dmaStoreCastPairs() const;

    // Legality queries
    bool isTypeLegalForUnit(mlir::seki_hw::HardwareUnit unit, llvm::ArrayRef<mlir::Type> inputTypes,
                            llvm::ArrayRef<mlir::Type> outputTypes) const;
    bool isCastSupported(mlir::Type from, mlir::Type to) const;

private:
    int64_t scratchpadBytes, dramBytes, dmaAlignment, numScratchpadBanks;
    int64_t macArrayRows, macArrayCols, computeUnits, vectorRegisterFileBytes;
    int64_t isaVersion, maxTileId;

    mlir::seki_hw::MACArrayCapAttr   macArrayCap;
    mlir::seki_hw::VecEngineCapAttr  vecEngineCap;
    mlir::seki_hw::ScalarUnitCapAttr scalarUnitCap;
    mlir::seki_hw::DMAEngineCapAttr  dmaEngineCap;
};
} // namespace mlir::seki
#endif // SEKI_TARGET_INFO_H
