#include "seki/SekiDialect.h"
#include "seki/SekiPasses.h"
#include "seki_hw/SekiHwDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<mlir::seki::SekiDialect>();
    registry.insert<mlir::seki_hw::SekiHwDialect>();
    mlir::registerAllPasses();
    registerSekiPasses();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "seki-opt", registry));
}
