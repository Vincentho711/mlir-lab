#include "seki/SekiTargetInfo.h"
#include "seki/SekiAttrs.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::seki {
SekiTargetInfo::SekiTargetInfo(mlir::ModuleOp mod) {
    auto target = mod->getAttrOfType<SekiTargetAttr>("seki.target");
    if (!target)
        llvm::report_fatal_error("module has no #seki.target attribute -"
                                 "pass --seki-arget to seki-opt");

    SekiMemoryConfigAttr mem = target.getMemory();
    scratchpadBytes = mem.getScratchpadBytes();
    dramBytes = mem.getDramBytes();
    dmaAlignment = mem.getDmaAlignment();
    numScratchpadBanks = mem.getNumScratchpadBanks();
}

int64_t SekiTargetInfo::getScratchpadBytes() const { return scratchpadBytes; }
int64_t SekiTargetInfo::getDRAMBytes() const { return dramBytes; }
int64_t SekiTargetInfo::getDMAAlignment() const { return dmaAlignment; }
int64_t SekiTargetInfo::getNumScratchpadBanks() const { return numScratchpadBanks; }

} // namespace mlir::seki
