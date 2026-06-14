#include "seki/SekiPasses.h"
#include "seki/SekiAttrs.h"
#include "seki/SekiTargetInfo.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#define GEN_PASS_DEF_MEMORYSPACEASSIGNMENTPASS
#include "seki/SekiPasses.h.inc"

namespace {

struct MemorySpaceAssignmentPass : impl::MemorySpaceAssignmentPassBase<MemorySpaceAssignmentPass> {

    using MemorySpaceAssignmentPassBase::MemorySpaceAssignmentPassBase;

    void runOnOperation() override {
        // Instantiate the SekiTargetInfo object
        mlir::ModuleOp mod = getOperation();

        // Running "--seki-assign-memory-spaces" without "--seki-attach-target" should error out gracefully
        if (!mod->hasAttr("seki.target")) {
            mod.emitError("seki-assign-memory-spaces requires "
                          "--seki-attach-target to run first");
            return signalPassFailure();
        }
        mlir::seki::SekiTargetInfo info(mod);
        int64_t scratchpadBytes = info.getScratchpadBytes();
        int64_t dmaAlignment = info.getDMAAlignment();

        mlir::MLIRContext *ctx = mod.getContext();
        mlir::Attribute scratchpadSpace =
            mlir::seki::MemorySpaceAttr::get(ctx, mlir::seki::MemorySpace::Scratchpad);

        // Depth-first traveral of the entire IR tree.
        // For every memref.alloc op anywhere inside the module, call this lambda with a typed handle to that op 
        // Note: promotes each allocation independently against full scratchpad budget.
        // It does not account for concurrent liveness, a proper bin-packing allocator with liveness analysis will be deferred. 
        mod.walk([&](mlir::memref::AllocOp alloc) {
            mlir::MemRefType type = alloc.getType();
            // MemRefType encodes everything that appears between the angle brackets in memref<4x3xf32, #seki.memory_space<...>>
            // E.g. type.getShape() -> ArrayRef<int64_t>{4, 3}
            // E.g. type.getElementType() -> DRAM/scratchpad

            // If it returns nullptr, then it is default 0 (DRAM).
            // If it is already defined with 1 (scratchpad), then no need to promote
            if (type.getMemorySpace()) return;
            // Check if it has dynamic shape like <?,?>
            // If so, we cannot prmote to scratchpad if dims are not known at compile-time
            if (!type.hasStaticShape()) return;

            // Calculate the number of bytes for this memref
            // getNumElements() returns 4*3=12 
            // getElementTypeBitWidth() returns the size of one element in bits, f32 would be 32. Divide it by 8 to get the byte per element. 
            int64_t totalBytes = type.getNumElements() * (type.getElementTypeBitWidth() / 8);
            // Do not promote if it won't be in scratchpad
            if (totalBytes > scratchpadBytes)
                return;

            // Convert memref<4x3xf32> to memref<4x3xf32, 1> automatically if it fits in scratchpad
            mlir::MemRefType newType = mlir::MemRefType::get(
                type.getShape(),
                type.getElementType(),
                type.getLayout(), // affine map. For exmaple, strided[1,4] (column major)
                scratchpadSpace
            );

            // DMA engine can only read from and write to addresses that are multiple of its alignment boundary.
            // Hence every scratchpad buffer must start at an address divisble by 64. 
            // By specifying the alignment attribute of memref.alloc, it tells lowering passes to allocate the buffer at an aligned address.
            // Without alignAttr, the alloc would get a random address and DMA engine would fail.
            mlir::IntegerAttr alignAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), dmaAlignment);

            // Instantiate an OpBuilder to build the new memref Op
            mlir::OpBuilder builder(alloc);
            auto newAlloc = mlir::memref::AllocOp::create(
                builder, alloc.getLoc(), newType, alignAttr
            );

            // Replace all allow with newAlloc
            alloc.replaceAllUsesWith(newAlloc.getResult());
            alloc.erase();

        });
    }
};
} // namespace

namespace mlir::seki {

std::unique_ptr<mlir::Pass> createMemorySpaceAssignmentPass() {
    return std::make_unique<MemorySpaceAssignmentPass>();
}

} // namespace mlir::seki
