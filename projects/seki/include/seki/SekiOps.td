#ifndef SEKI_OPS_TD
#define SEKI_OPS_TD

include "seki/SekiDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

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
#endif // SEKI_OPS_TD
