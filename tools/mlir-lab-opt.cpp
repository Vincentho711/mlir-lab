#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Per-module registration pattern: each lib/ module exposes a
// registerPasses() entry point and is added as a dep in tools/BUILD.bazel.
// Example when lib/Transform/Affine is added:
//
//   #include "mlir_lab/Transform/Affine/Passes.h"
//   ...
//   mlir::lab::affine::registerPasses();

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "MLIR Lab Optimizer", registry));
}
