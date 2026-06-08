#ifndef SEKI_OPS_TD
#define SEKI_OPS_TD

include "seki/SekiDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

//===-----------------------------------------------===//
// Seki Op base class
//===-----------------------------------------------===//
class SekiOp<string mnemonic, list<Trait> traits = []>
    : Op<SekiDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===/
// Pure marks the op has no memory side effects, result is entirely determined by the inputs.
def Seki_ReluOp : SekiOp<"relu", [Pure, SameOperandsAndResultType]> {
    let summary = "Element-wise ReLU activation: max(0, x)";
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
    let assemblyFormat = "$input attr-dict `:` type($input)";
    let hasVerifier = 1;
}

def Seki_MatmulOp : SekiOp<"matmul", [DeclareOpInterfaceMethods<DestinationStyleOpInterface>, AllTypesMatch<["C", "result"]>]> {
    let summary = "Matrix multiply-accumulate: C_out = A * B + C_in";
    let arguments = (ins
        AnyRankedTensor:$A,
        AnyRankedTensor:$B,
        AnyRankedTensor:$C
    );
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{
        `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
        `outs` `(` $C `:` type($C) `)` attr-dict `->` type($result)
    }];
    let hasVerifier = 1;
}
#endif // SEKI_OPS_TD
