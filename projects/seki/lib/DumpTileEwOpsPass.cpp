#include "seki_hw/SekiHwPasses.h"
#include "seki_hw/SekiHwDialect.h"
#include "seki_hw/SekiHwInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

#define GEN_PASS_DEF_DUMPTILEEWOPSPASS
#include "seki_hw/SekiHwPasses.h.inc"

namespace {
struct DumpTileEwOpsPass : impl::DumpTileEwOpsPassBase<DumpTileEwOpsPass> {
    // Inherits constructor from the base, which is what the generated createDumpTileEwOpsPass() factory calls internally.
    using DumpTileEwOpsPassBase::DumpTileEwOpsPassBase;
    void runOnOperation() override {
        getOperation().walk([](mlir::Operation *op) {
            // Try to cast the op to see if it has SekiTileEwOpInterface
            auto iface = mlir::dyn_cast<mlir::seki_hw::SekiTileEwFnOpInterface>(op);
            // If not, simply return
            if (!iface)
                return;
            // If so, emit a remark and print the op name
            mlir::Attribute fn = iface.getEwFn(); // Should be either TileEwBinaryFn or TileEwUnaryFn
            op->emitRemark() << op->getName() << ": fn = " << fn; 
        });
    }
};
}
