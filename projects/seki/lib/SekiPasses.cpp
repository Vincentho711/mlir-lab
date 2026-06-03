#include "seki/SekiDialect.h"
#include "seki/SekiAttrs.h"
#include "seki/SekiTargets.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MemoryBuffer.h"

static llvm::cl::opt<std::string> sekiTargetOption(
    "seki-target",
    llvm::cl::desc("Seki hardware target: named key (e.g. seki-v1) "
                   "or path to a .mlir attribute file"),
    llvm::cl::init(""));

namespace {

static bool isFilePath(llvm::StringRef s) {
    return s.ends_with(".mlir") || s.starts_with("/") ||
           s.starts_with("./")  || s.starts_with("../");
}

struct SekiAttachTargetPass
    : mlir::PassWrapper<SekiAttachTargetPass,
                        mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SekiAttachTargetPass)

    llvm::StringRef getArgument()    const override { return "seki-attach-target"; }
    llvm::StringRef getDescription() const override {
        return "Resolve --seki-target and attach #seki.target<...> to the module";
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::seki::SekiDialect>();
    }

    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();

        if (sekiTargetOption.empty()) {
            mod.emitError("--seki-target not set; "
                          "pass --seki-target=seki-v1 or a path to a .mlir file");
            return signalPassFailure();
        }

        mlir::Attribute attr;

        if (isFilePath(sekiTargetOption)) {
            // External prototype target - load from file
            auto buf = llvm::MemoryBuffer::getFile(sekiTargetOption);
            if (!buf) {
                mod.emitError("cannot open target file: ") << sekiTargetOption;
                return signalPassFailure();
            }
            attr = mlir::parseAttribute((*buf)->getBuffer(), mod.getContext());
            if (!attr) {
                mod.emitError("failed to parse target attribute from: ")
                    << sekiTargetOption;
                return signalPassFailure();
            }
        } else {
            // Built-in named target - construct via factory.
            attr = mlir::seki::getBuiltinTarget(sekiTargetOption, mod.getContext());
            if (!attr) {
                mod.emitError("unknown target '") << sekiTargetOption
                    << "'; known built-in targets: seki-v1";
                return signalPassFailure();
            }
        }

        mod->setAttr("seki.target", attr);
    }
};

} // namespace

namespace mlir::seki {

void registerMemorySpaceAssignmentPass();

void registerSekiPasses() {
    mlir::PassRegistration<SekiAttachTargetPass>();
    registerMemorySpaceAssignmentPass();
}

} // namespace mlir::seki
