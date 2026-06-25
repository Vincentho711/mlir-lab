#include "seki_hw/SekiHwPasses.h"
#include "seki_hw/SekiHwDialect.h"
#include "seki_hw/SekiHwTypes.h"
#include "seki/SekiTargetInfo.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#define GEN_PASS_DEF_SEKIHWVERIFYTARGETPASS
#include "seki_hw/SekiHwPasses.h.inc"

namespace {

constexpr int64_t kExpectedISAVersion = 1;

struct SekiHwVerifyTargetPass : impl::SekiHwVerifyTargetPassBase<SekiHwVerifyTargetPass> {
    using SekiHwVerifyTargetPassBase::SekiHwVerifyTargetPassBase;
    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        mlir::seki::SekiTargetInfo info(module);

        // Check ISA version to ensure hardware and compiler is in sync
        if (info.getISAVersion() != kExpectedISAVersion) {
            module.emitError() << "ISA version mismatch: target has version "
                               << info.getISAVersion() << ", compiler expects " << kExpectedISAVersion;
            return signalPassFailure();
        }

        // Check vec register file usage (walk FuncOps)
        module.walk([&](mlir::func::FuncOp func){
            checkVecRegisterFile(func, info);
        });

        // Check tile usage (walk FuncOps)
        module.walk([&](mlir::func::FuncOp func){
            checkTileSlotDemand(func, info);
        });
    }
private:
    void checkVecRegisterFile(mlir::func::FuncOp func, const mlir::seki::SekiTargetInfo &info);
    void checkTileSlotDemand(mlir::func::FuncOp func, const mlir::seki::SekiTargetInfo &info);
};

void SekiHwVerifyTargetPass::checkVecRegisterFile(mlir::func::FuncOp func, const mlir::seki::SekiTargetInfo &info) {
    mlir::Liveness liveness(func);
    bool failed = false;

    std::function<void(mlir::Region &)> checkRegion = [&](mlir::Region &region) {
        for (mlir::Block &block : region) {
            if (failed) return;

            // Get the precomputed liveness info for this block
            const mlir::LivenessBlockInfo *blockInfo = liveness.getLiveness(&block);
            if (!blockInfo) continue; // unreachable block

            // Seed with vec values already live at block entry before the first op of this block executes.
            // liveVecs is created once per block and mutated in place as the op loop advances.
            llvm::SmallPtrSet<mlir::Value, 8> liveVecs;
            // blockInfo->in contains all values live at block entry - values defined in predecessor blocks that are still in here.
            for (mlir::Value v : blockInfo->in())
                // We filter to only VecType values since tiles and scalars don't count against the vec register file budget.
                if (mlir::isa<mlir::seki_hw::VecType>(v.getType()))
                    liveVecs.insert(v);

            // For each op in the block, add its vec-typed results to the live set.
            for (mlir::Operation &op : block) {
                // Add vec results defined by this op
                for (mlir::Value result : op.getResults())
                    if (mlir::isa<mlir::seki_hw::VecType>(result.getType()))
                        liveVecs.insert(result);

                // Sum live bytes across all simultaneously live vecs
                int64_t liveBytes = 0;
                for (mlir::Value v : liveVecs) {
                    auto vecType = mlir::cast<mlir::seki_hw::VecType>(v.getType());
                    // getVecRegisterSizeBytes(elemType) returns macArrayCols * sizeof(elemType) - i.e. 128 * 4 = 512 for f32,
                    // 128 * 2 = 256 for bf16. Each value in liveVecs contributes its own type-specific cost.
                    liveBytes += info.getVecRegisterSizeBytes(vecType.getElementType());
                }
                // If the sum exceeds the maximum vector register file bytes, emit an error, then set failed and return to stop processing.
                if (liveBytes > info.getVectorRegisterFileBytes()) {
                    op.emitError() << "vec register file overflow: "
                                   << liveBytes << " bytes live, hardware limit is "
                                   << info.getVectorRegisterFileBytes() << " bytes";
                    signalPassFailure();
                    failed = true;
                    return;
                }

                // After checking, release values whose live range ends at this op.
                // getEndOperation(v, startOp) returns the last op in this block where v is still needed. Once that has executed, the register is free.
                llvm::SmallVector<mlir::Value> toRemove;
                for (mlir::Value v : liveVecs) {
                    // startOp locates the op across block
                    // If checks whether the defining op belongs to the same block, if so, set the startOp as the defining op, else the op was defined
                    // in predeceesor block, return the startOp as the start of this block
                    // E.g. 
                    // predecessor block:
                    //   %v = tile_reduce_vec ...
                    // this block:              <- v is live-in, startOp = &block.front()
                    //   op A: vec_ew_binary %v, ... <- getEndOperation find op A as last use
                    //   op B: ...
                    //   op C: return
                    mlir::Operation *startOp = (v.getDefiningOp() && v.getDefiningOp()->getBlock() == &block) ?
                    v.getDefiningOp() : &block.front();
                    if (blockInfo->getEndOperation(v, startOp) == &op)
                        toRemove.push_back(v);
                }
                // Remove the unused vec
                for (mlir::Value v : toRemove)
                    liveVecs.erase(v);

                // Recurse into nested regions (e.g. scf.for body)
                for (mlir::Region &nested : op.getRegions())
                    checkRegion(nested);
            }
        }
    };
    checkRegion(func.getBody());

}

void SekiHwVerifyTargetPass::checkTileSlotDemand(mlir::func::FuncOp func, const mlir::seki::SekiTargetInfo &info) {
    // Compute liveness analysis on the module such that by the time we iterate blocks, all the live-in/live-out sets are already precomputed.
    mlir::Liveness liveness(func);

    std::function<void(mlir::Region &)> checkRegion = [&](mlir::Region &region) {
        for (mlir::Block &block : region) {
            const mlir::LivenessBlockInfo *blockInfo = liveness.getLiveness(&block);
            if (!blockInfo) continue;

            // liveTiles is created once per block and mutated in place as the op loop advances
            llvm::SmallPtrSet<mlir::Value, 16> liveTiles;
            // E.g.
            // block 0 (entry)
            // ------------------------------------------------------
            // %tile1 = seki_hw.load %addr1 -> !seki_hw.tile<f32> 
            // %tile2 = seki_hw.load %addr2 -> !seki_hw.tile<f32> 
            // cf.br ^block1
            // ------------------------------------------------------
            // block1 (successor of block0)
            // ------------------------------------------------------
            // ^block1:
            //    %tile3 = seki_hw.tile_ew_binary<add> %tile1, %tile2 : !seki_hw.tile<f32>
            //    return %tile3 : !seki_hw.tile<f32>
            // Here, tile1, tile2 is used in block1.
            // Even though liveTiles is created fresh in every block, the newly created liveTiles is seeded from blockInfo->info
            // for block1. The liveness analysis knows tile1 and tile2 are live-in to block1 (they were porduced in block0 and used in block1) , so:
            // seed: liveTiles = {tile1, tile2}  <- at the start of block1
            for (mlir::Value t : blockInfo->in())
                if (mlir::isa<mlir::seki_hw::TileType>(t.getType()))
                    liveTiles.insert(t);

            for (mlir::Operation &op : block) {
                // Add Tile results defined by this op
                for (mlir::Value result : op.getResults())
                    if (mlir::isa<mlir::seki_hw::TileType>(result.getType()))
                        liveTiles.insert(result);

                // Sum live tiles across all simultaneously live tiles
                int64_t liveTileCount = static_cast<int64_t>(liveTiles.size());
                // If the sum exceeds the maximum tile slots, emit an warning. Should move to error later.
                if (liveTileCount > info.getMaxTileSlots()) {
                    op.emitWarning() << "peak live tile count "
                                     << liveTileCount << " exceeds hardware slot limit of "
                                     << info.getMaxTileSlots() ;
                }

                // After checking, release values whose live range ends at this op.
                // getEndOperation(v, startOp) returns the last op in this block where v is still needed. Once that has executed, the register is free.
                llvm::SmallVector<mlir::Value> toRemove;
                for (mlir::Value v : liveTiles) {
                    mlir::Operation *startOp = (v.getDefiningOp() && v.getDefiningOp()->getBlock() == &block) ?
                    v.getDefiningOp() : &block.front();
                    if (blockInfo->getEndOperation(v, startOp) == &op)
                        toRemove.push_back(v);
                }
                // Remove the unused tiles
                for (mlir::Value v : toRemove)
                    liveTiles.erase(v);

                // Recurse into nested regions (e.g. scf.for body)
                for (mlir::Region &nested : op.getRegions())
                    checkRegion(nested);
            }
        }
    };
    checkRegion(func.getBody());
}
} // namespace
