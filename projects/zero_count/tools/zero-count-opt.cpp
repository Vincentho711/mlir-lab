#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "zero_count/ZeroCountDialect.h"

// Forward declarations
namespace mlir::zero_count { void registerZeroCountPasses(); }

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;

    // Register all built-in MLIR dialects and passes so the standard
    // lowering flags (e.g. --convert-arith-to-llvm) still work.
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();

    // Reigster zero_count dialect into the same registry.
    registry.insert<mlir::zero_count::ZeroCountDialect>();

    // Register custom passes
    mlir::zero_count::registerZeroCountPasses();

    return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "zero-count-opt", registry));
}
