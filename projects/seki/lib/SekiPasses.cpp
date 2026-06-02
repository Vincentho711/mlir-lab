#include "seki/SekiDialect.h"
#include "seki/SekiAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"

static llvm::cl::opt<std::string> sekiTargetOption(
    "seki-target",
    llvm::cl::desc("Seki hardware target: named key (e.g. seki-v1) "
                   "or path to a .mlir attribute file"),
    llvm::cl::init(""));

namespace {

// Maps short target names to their canonical .mlir file paths.
// Paths are relative to the workspace root — valid for `bazel run`.
// TODO: Replace relative paths with Bazel runfiles data deps or
// build-time genrule embedding so installed binaries can locate target files
// without requiring the binary to run from the workspace root.
static const llvm::StringMap<std::string> &namedTargetRegistry() {
    static llvm::StringMap<std::string> registry{
        {"seki-v1", "projects/seki/targets/seki-v1.mlir"},
    };
    return registry;
}

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

        // Resolve named target to its file path.
        std::string filePath;
        if (isFilePath(sekiTargetOption)) {
            filePath = sekiTargetOption;
        } else {
            const auto &reg = namedTargetRegistry();
            auto it = reg.find(sekiTargetOption);
            if (it == reg.end()) {
                mod.emitError("unknown target '") << sekiTargetOption
                    << "'; known targets: seki-v1";
                return signalPassFailure();
            }
            filePath = it->second;
        }

        // Load and parse the target file.
        auto buf = llvm::MemoryBuffer::getFile(filePath);
        if (!buf) {
            mod.emitError("cannot open target file: ") << filePath;
            return signalPassFailure();
        }

        mlir::Attribute attr =
            mlir::parseAttribute((*buf)->getBuffer(), mod.getContext());
        if (!attr) {
            mod.emitError("failed to parse target attribute from: ") << filePath;
            return signalPassFailure();
        }

        mod->setAttr("seki.target", attr);
    }
};

} // namespace

namespace mlir::seki {

void registerSekiPasses() {
    mlir::PassRegistration<SekiAttachTargetPass>();
}

} // namespace mlir::seki
