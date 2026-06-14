#include "seki/SekiPasses.h"
#include "seki/SekiAttrs.h"
#include "seki/SekiDialect.h"
#include "seki/SekiTargets.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "llvm/Support/MemoryBuffer.h"

#define GEN_PASS_DEF_SEKIATTACHTARGETPASS
#include "seki/SekiPasses.h.inc"

namespace {

static bool isFilePath(llvm::StringRef s) {
    return s.ends_with(".mlir") || s.starts_with("/") ||
           s.starts_with("./")  || s.starts_with("../");
}

struct SekiAttachTargetPass
    : impl::SekiAttachTargetPassBase<SekiAttachTargetPass> {

    using SekiAttachTargetPassBase::SekiAttachTargetPassBase;

    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();

        if (target.empty()) {
            mod.emitError("--seki-target not set; "
                          "pass --seki-target=seki-v1 or a path to a .mlir file");
            return signalPassFailure();
        }

        mlir::Attribute attr;

        if (isFilePath(target)) {
            // External prototype target - load from file
            auto buf = llvm::MemoryBuffer::getFile(target);
            if (!buf) {
                mod.emitError("cannot open target file: ") << target;
                return signalPassFailure();
            }
            attr = mlir::parseAttribute((*buf)->getBuffer(), mod.getContext());
            if (!attr) {
                mod.emitError("failed to parse target attribute from: ") << target;
                return signalPassFailure();
            }
        } else {
            // Built-in named target - construct via factory.
            attr = mlir::seki::getBuiltinTarget(target, mod.getContext());
            if (!attr) {
                mod.emitError("unknown target '") << target
                    << "'; known built-in targets: seki-v1";
                return signalPassFailure();
            }
        }

        mod->setAttr("seki.target", attr);
    }
};

} // namespace

namespace mlir::seki {

std::unique_ptr<mlir::Pass> createSekiAttachTargetPass() {
    return std::make_unique<SekiAttachTargetPass>();
}

} // namespace mlir::seki
