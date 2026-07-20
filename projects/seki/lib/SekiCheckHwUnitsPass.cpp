#include "seki/SekiPasses.h"
#include "seki/SekiTargetInfo.h"  // SekiTargetInfo — also pulls in SekiHwAttrs.h → stringifyHardwareUnit
#include "seki/SekiOps.h"         // CastOp, SekiExecUnitOpInterface
#include "seki/SekiAttrs.h"       // SekiTargetAttr
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#define GEN_PASS_DEF_SEKICHECKHWUNITSPASS
#include "seki/SekiPasses.h.inc"

namespace {

// MatmulOp example:
// %r = seki.matmul ins(%A, %B: tensor<256x512xbf16>, tensor<512x128xbf16>)
//                  outs(%C: tensor<256x128xf32>) -> tensor<256x128xf32>
// MatmulOp implements DestinationStyleOpInterface (DPS) which labels C as a "DPS init".
// getDpsInitsMutable() reutrns operand index 2 with length 1.
// C should be treated as a result rather than input, otherwise check would fail
// Only pass A and B as inputs types, C's type is checked via the result
static llvm::SmallVector<mlir::Type> getInsElementTypes(mlir::Operation *op) {
    llvm::SmallVector<mlir::Type> types;

    if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op)) {
        // Iterate through all operands of the op with DPS
        for (mlir::OpOperand &operand : op->getOpOperands()) {
            // Check if an operand is a DPS
            // A and B are not, but C is.
            if (!dps.isDpsInit(&operand)) // Skip C
                // Only push non-dps operands into the types vector
                types.push_back(mlir::cast<mlir::RankedTensorType>(operand.get().getType()).getElementType());
        }
    } else {
        // Non DPS op (EwUnaryOp, EwBinaryOp, ...) : every operand is an input
        for (mlir::Value v : op->getOperands()) {
            if (auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType()))
              types.push_back(t.getElementType());
        }
    }
    return types;
}

// Returns the element type of every result tensor.
// Op results are true outputs irrespective of DPS
static llvm::SmallVector<mlir::Type> getResultElementTypes(mlir::Operation *op) {
    llvm::SmallVector<mlir::Type> types;
    for (mlir::Value v : op->getResults())
        if (auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType()))
            types.push_back(t.getElementType());
    return types;
}

struct SekiCheckHwUnitsPass : impl::SekiCheckHwUnitsPassBase<SekiCheckHwUnitsPass> {
    using SekiCheckHwUnitsPassBase::SekiCheckHwUnitsPassBase;

    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();
        // Check whether SekiTargetAttr has seki.target attribute
        if (!mod->getAttrOfType<mlir::seki::SekiTargetAttr>("seki.target"))
            return;
        mlir::seki::SekiTargetInfo info(mod);
        bool failed = false;

        mod.walk([&](mlir::Operation *op) {
            // seki.cast doesn't implement SekiExecUnitOpInterface as it might be fused into concrete seki_hw depending on precedent op
            // Handle separetely, we check whether this cast can be supported by seki vec engine, dma store and dma load.
            if (auto cast = mlir::dyn_cast<mlir::seki::CastOp>(op)) {
                auto in = mlir::cast<mlir::RankedTensorType>(cast.getInput().getType());
                auto out = mlir::cast<mlir::RankedTensorType>(cast.getOutput().getType());
                if (!info.isCastSupported(in.getElementType(), out.getElementType())) {
                    op->emitError() << "no cast path (Vec or DMA) supports "
                                    << in.getElementType() << " -> "
                                    << out.getElementType() << " on this target";
                    failed = true;
                }
                return;
            }

            // Check ops that implement the SekiExecUnitOpInterface.
            auto iface = mlir::dyn_cast<mlir::seki::SekiExecUnitOpInterface>(op);
            if (!iface) return;

            // Get a list of all input operands types, handle DPS ins, ignore them as input.
            llvm::SmallVector<mlir::Type> insTypes = getInsElementTypes(op);
            // Get a list of all output operands types, handle DPS ins and treat them as output.
            llvm::SmallVector<mlir::Type> outTypes = getResultElementTypes(op);

            // Iterate through all the required units, and ensure that each unit can process the input and output types.
            for (mlir::seki_hw::HardwareUnit unit : iface.getRequiredUnits()) {
                // Check 1: does the target even have this unit?
                if (!info.hasUnit(unit)) {
                    op->emitError() << "requires "
                                    << mlir::seki_hw::stringifyHardwareUnit(unit)
                                    << ", but target has none";
                    failed = true;
                    return;
                }
                // Check 2: are the op's element types legal for that unit
                if (!info.isTypeLegalForUnit(unit, insTypes, outTypes)) {
                    op->emitError() << mlir::seki_hw::stringifyHardwareUnit(unit)
                                    << " does not support these types on this target; "
                                    << "did you mean to run --seki-cast-to-target-types?";
                    failed = true;
                }
            }
        });
    }
};
} // namespace
