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

#endif // SEKI_INTERFACES_TD
