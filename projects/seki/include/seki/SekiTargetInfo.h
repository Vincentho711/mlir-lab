#ifndef SEKI_TARGET_INFO_H
#define SEKI_TARGET_INFO_H

#include "mlir/IR/BuiltinOps.h"

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

private:
    int64_t scratchpadBytes, dramBytes, dmaAlignment, numScratchpadBanks;
    int64_t macArrayRows, macArrayCols, computeUnits, vectorRegisterFileBytes;
    int64_t isaVersion, maxTileId;
};
} // namespace mlir::seki
#endif // SEKI_TARGET_INFO_H
