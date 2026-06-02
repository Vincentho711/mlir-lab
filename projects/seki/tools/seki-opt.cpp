#include "seki/SekiDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir::seki { void registerSekiPasses(); }

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<mlir::seki::SekiDialect>();
    mlir::registerAllPasses();
    mlir::seki::registerSekiPasses();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "seki-opt", registry));
}
