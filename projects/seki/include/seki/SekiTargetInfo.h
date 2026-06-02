#ifndef SEKI_TARGET_INFO_H
#define SEKI_TARGET_INFO_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::seki {

class SekiTargetInfo {
public:
    explicit SekiTargetInfo(mlir::ModuleOp mod);

    int64_t getScratchpadBytes() const;
    int64_t getDRAMBytes() const;
    int64_t getDMAAlignment() const;
    int64_t getNumScratchpadBanks() const;
private:
    int64_t scratchpadBytes;
    int64_t dramBytes;
    int64_t dmaAlignment;
    int64_t numScratchpadBanks;
};
} // namespace mlir::seki
#endif // SEKI_TARGET_INFO_H
