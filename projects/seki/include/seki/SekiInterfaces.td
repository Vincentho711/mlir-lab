#ifndef SEKI_INTERFACES_TD
#define SEKI_INTERFACES_TD

include "mlir/IR/OpBase.td"

def SekiEwFnOpInterface : OpInterface<"SekiEwFnOpInterface"> {
    let description = [{
        Interface implemented by seki elementwise ops that carry a fn enum attribute (ew_unary, ew_binary).
        Allows passes to uniformly detect any elementwise op without switching.
    }];
    let cppNamespace = "::mlir::seki";
    let methods = [
        InterfaceMethod<
            "Return the function attribute (UnaryFnAttr or BinaryFnAttr)",
            "::mlir::Attribute",
            "getEwFn",
            (ins),
            [{}],
            [{ return $_op.getFnAttr(); }]
        >,
    ];
}

def SekiExecUnitOpInterface : OpInterface<"SekiExecUnitOpInterface"> {
    let description = [{
        Interface implemented by all seki ops that require specific hardware execution units.
        Returns all units the op needs, every listed unit must be present on the target for op to be legal.
        Allows passes to query this interface to validate model-hardware compatability.
    }];
    let cppNamespace = "::mlir::seki";
    let methods = [
        InterfaceMethod<
            "Hardware execution units this op requires, all of them must be present on the target",
            "::llvm::SmallVector<::mlir::seki_hw::HardwareUnit>",
            "getRequiredUnits",
            (ins)
        >,
    ];
}

#endif // SEKI_INTERFACES_TD
