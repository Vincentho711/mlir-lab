#include "seki/SekiPasses.h"
#include "seki/SekiTargetInfo.h"
#include "seki/SekiOps.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/BuiltinOps.h"

#define GEN_PASS_DEF_SEKITILEOPSPASS
#include "seki/SekiPasses.h.inc"

namespace {

struct SekiTileOpsPass : impl::SekiTileOpsPassBase<SekiTileOpsPass> {
    using SekiTileOpsPassBase::SekiTileOpsPassBase;
    
    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();
        if (!mod->getAttrOfType<mlir::seki::SekiTargetAttr>("seki.target"))
            return;
        mlir::seki::SekiTargetInfo info(mod);
        if (!info.hasMACArray())
            return;
        int64_t tileRows = info.getMACRows();
        int64_t tileCols = info.getMACCols();

        // Collect all ops that support tiling.
        llvm::SmallVector<mlir::TilingInterface> toTile;
        mod.walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir::seki::MatmulOp, mlir::seki::BatchMatmulOp>(op)) 
                toTile.push_back(mlir::cast<mlir::TilingInterface>(op));
        });

        mlir::IRRewriter rewriter(&getContext());
        for (mlir::TilingInterface op : toTile) {
            // Tile sizes mirror the iteration domain: [M, K] for seki.matmul.
            // [batch, M, K] for seki.batch_matmul.
            llvm::SmallVector<mlir::OpFoldResult> tileSizes;
            if (mlir::isa<mlir::seki::BatchMatmulOp>(op.getOperation()))
                tileSizes.push_back(rewriter.getIndexAttr(1));
            tileSizes.push_back(rewriter.getIndexAttr(tileRows));
            tileSizes.push_back(rewriter.getIndexAttr(tileCols));

            mlir::scf::SCFTilingOptions options;
            options.setTileSizes(tileSizes);

            rewriter.setInsertionPoint(op);
            mlir::FailureOr<mlir::scf::SCFTilingResult> tilingResult =
                mlir::scf::tileUsingSCF(rewriter, op, options);
            if (mlir::failed(tilingResult)) {
                op.getOperation()->emitError("failed to tile");
                signalPassFailure();
                return;
            }
            rewriter.replaceOp(op, tilingResult->replacements);
        }
    }
};
} // namespace
