#ifndef SEKI_HW_INTERFACES_TD
#define SEKI_HW_INTERFACES_TD

include "mlir/IR/OpBase.td"

def SekiTileEwOpInterface : OpInterface<"SekiTileEwOpInterface"> {
    let description = [{
        Interface implemented by all tile elementwise ops (tile_ew_unary, tile_ew_binary).
        Allows passes to iterate uniformly over tile EW ops without switching on concrete op types.
    }];
    let cppNamespace = "::mlir::seki_hw";
    let methods = [
        InterfaceMethod<
            "Return the elementwise function attribute (UnaryFnAttr or BinaryFnAttr)", // doc
            "::mlir::Attribute", // return_type, TileEwUnaryOp::getFn() return UnaryFnAttr and TileEwBinaryOp::getFn() returns BinaryFnAttr, both are EnumAttr which is mlir::Attribute
            "getEwFn", // method_name
            (ins), // args, no arguments beyond the implicit op itself
            [{}], // extraClassOf
            [{ return $_op.getFnAttr(); }] // default_impl, $_op is tblgen's placeholder for concrete op instance. Both TileEwUnaryOp and TileEwBinaryOp have a getFnAttr() method generated from their fn argument.
        >,
    ];
}

def SekiVecEwOpInterface : OpInterface< "SekiVecEwOpInterface"> {
    let description = [{
        Interface implemented by all vec elementwise ops (vec_ew_unary, vec_ew_binary).
        Allows passes to iterate uniformly over vec EW ops without switching on concrete op types.
    }];
    let cppNamespace = "::mlir::seki_hw";
    let methods = [
        InterfaceMethod<
            "Return the elementwise function attribute (UnaryFnAttr or BinaryFnAttr)", // doc
            "::mlir::Attribute", // return_type, TileEwUnaryOp::getFn() return UnaryFnAttr and TileEwBinaryOp::getFn() returns BinaryFnAttr, both are EnumAttr which is mlir::Attribute
            "getEwFn", // method_name
            (ins), // args, no arguments beyond the implicit op itself
            [{}], // extraClassOf
            [{ return $_op.getFnAttr(); }] // default_impl, $_op is tblgen's placeholder for concrete op instance. Both TileEwUnaryOp and TileEwBinaryOp have a getFnAttr() method generated from their fn argument.
        >,
    ];
}
#endif // SEKI_HW_INTERFACES_TD
