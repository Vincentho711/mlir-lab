#ifndef SEKI_OPS_TD
#define SEKI_OPS_TD

include "seki/SekiDialect.td"
include "seki/SekiAttrs.td"
include "seki/SekiInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/TilingInterface.td"

//===-----------------------------------------------===//
// Seki Op base class
//===-----------------------------------------------===//
class SekiOp<string mnemonic, list<Trait> traits = []>
    : Op<SekiDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===/
def Seki_MatmulOp : SekiOp<"matmul", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
    DeclareOpInterfaceMethods<SekiExecUnitOpInterface>,
    DeclareOpInterfaceMethods<TilingInterface,
        ["getIterationDomain", "getLoopIteratorTypes",
         "getResultTilePosition", "getTiledImplementation"]>,
    SekiTileableOpInterface,
    AllTypesMatch<["C", "result"]>]> {
    let summary = "Matrix multiply-accumulate: C_out = A * B + C_in";
    let arguments = (ins
        AnyRankedTensor:$A,
        AnyRankedTensor:$B,
        AnyRankedTensor:$C,
        OptionalAttr<UnitAttr>:$transpose_b
    );
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{
        `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
        `outs` `(` $C `:` type($C) `)` attr-dict `->` type($result)
    }];
    let hasVerifier = 1;
}

def Seki_EwUnaryOp : SekiOp<"ew_unary",
    [Pure, SameOperandsAndResultType, SekiEwFnOpInterface, SekiEpilogueFusableOpInterface,
     DeclareOpInterfaceMethods<SekiExecUnitOpInterface>]> {
    let summary = "Unary elementwise op on a tensor";
    let arguments = (ins Seki_UnaryFnAttr:$fn, AnyTensor:$input);
    let results   = (outs AnyTensor:$output);
    let assemblyFormat = "`<` $fn `>` $input attr-dict `:` type($input)";
    let hasVerifier = 1;
}

def Seki_EwBinaryOp : SekiOp<"ew_binary",
    [Pure, SameOperandsAndResultType, SekiEwFnOpInterface, SekiEpilogueFusableOpInterface,
     DeclareOpInterfaceMethods<SekiExecUnitOpInterface>]> {
    let summary = "Binary elementwise op on tensors";
    let arguments = (ins Seki_BinaryFnAttr:$fn, AnyTensor:$lhs, AnyTensor:$rhs);
    let results   = (outs AnyTensor:$output);
    let assemblyFormat = "`<` $fn `>` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def Seki_ReduceOp : SekiOp<"reduce", [Pure, DeclareOpInterfaceMethods<SekiExecUnitOpInterface>]> {
    let summary = "Reduce tensor along one axis";
    let arguments = (ins Seki_ReduceFnAttr:$fn, AnyRankedTensor:$input, I64Attr:$axis);
    let results   = (outs AnyRankedTensor:$output);
    let assemblyFormat = "`<` $fn `>` $input `axis` `=` $axis attr-dict `:` type($input) `->` type($output)";
    let hasVerifier = 1;
}

def Seki_BatchMatmulOp : SekiOp<"batch_matmul", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>, 
    DeclareOpInterfaceMethods<SekiExecUnitOpInterface>, 
    DeclareOpInterfaceMethods<TilingInterface,
        ["getIterationDomain", "getLoopIteratorTypes",
         "getResultTilePosition", "getTiledImplementation"]>,
    SekiTileableOpInterface,
    AllTypesMatch<["C", "result"]>]> {
    let summary = "Batched matrix multiply-accumulate: C_out = A * B + C_in, with leading batch dim";
    let arguments = (ins
        AnyRankedTensor:$A,
        AnyRankedTensor:$B,
        AnyRankedTensor:$C,
        OptionalAttr<UnitAttr>:$transpose_b
    );
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{
        `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
        `outs` `(` $C `:` type($C) `)` attr-dict `->` type($result)
    }];
    let hasVerifier = 1;
}

def Seki_BroadcastOp : SekiOp<"broadcast", [Pure]> {
    let summary = "Broadcast tensor along specified dimensions";
    let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$broadcast_dims);
    let results   = (outs AnyRankedTensor:$output);
    let assemblyFormat = "$input `broadcast_dims` `=` $broadcast_dims attr-dict `:` type($input) `->` type($output)";
    let hasVerifier = 1;
}

def Seki_TransposeOp : SekiOp<"transpose", [Pure]> {
    let summary = "Permute tensor dimensions";
    let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$permutation);
    let results   = (outs AnyRankedTensor:$output);
    let assemblyFormat = "$input `permutation` `=` $permutation attr-dict `:` type($input) `->` type($output)";
    let hasVerifier = 1;
}

def Seki_ReshapeOp : SekiOp<"reshape", [Pure]> {
    let summary = "Reshape tensor to a new shape (same element count)";
    let arguments = (ins AnyRankedTensor:$input);
    let results = (outs AnyRankedTensor:$output);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
    let hasVerifier = 1;
}

def Seki_ExtractOp : SekiOp<"extract", [Pure]> {
    let summary = "Extract a row or column from a 2D tensor as a 1D tensor";
    let arguments = (ins Seki_AxisAttr:$axis, AnyRankedTensor:$input, Index:$idx);
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{`<` $axis `>` $input `,` $idx attr-dict `:` type($input) `,` type($idx) `->` type($result)}];
    let hasVerifier = 1;
}

def Seki_UpdateOp : SekiOp<"update", [Pure, AllTypesMatch<["dest", "result"]>]> {
    let summary = "Write a 1D tensor into a row or column of a 2D tensor (destination-style)";
    let arguments = (ins Seki_AxisAttr:$axis, AnyRankedTensor:$vec, Index:$idx, AnyRankedTensor:$dest);
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{`<` $axis `>` $vec `,` $idx `outs` `(` $dest `:` type($dest) `)` 
                           attr-dict `:` type($vec) `,` type($idx) `->` type($result)}];
    let hasVerifier = 1;
}

def Seki_CastOp: SekiOp<"cast", [Pure, SekiEpilogueFusableOpInterface]> {
    let summary = "Elementwise type conversion on a tensor (e.g. f32 -> bf16)";
    let arguments = (ins AnyRankedTensor:$input);
    let results = (outs AnyRankedTensor:$output);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
    let hasVerifier = 1;
}

def Seki_FusedComputationOp: SekiOp<"fused_computation", [
    Pure,
    // Region body can only reference its own block args and internal SSA.
    // External tensors must be passed as $inputs and mapped to block arguments. 
    IsolatedFromAbove,
    // Exactly one basic block, no branches or loops inside region.
    // Tiling loops live outside, hence region body is always straight line.
    SingleBlock
]> {
    let summary = "A fused group of ops sharing one scratchpad tile";
    let arguments = (ins Variadic<AnyTensor>:$inputs);
    let results = (outs AnyTensor:$output);
    // SizeRegion<1> enforces the SingleBlock constraint
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = [{
        `(` $inputs `:` type($inputs) `)` `->` type($output) $body attr-dict
    }];
    let hasVerifier = 1;
}

def Seki_YieldOp: SekiOp<"yield", [
    // Terminator: this op ends a basic block (like func.return or scf.yield).
    Terminator,
    // ReturnLike: tells the canonicalizer this op "returns" a value from its enclosing region
    // Enabling region-level folding and inlining optimisations.
    Pure,
    ReturnLike
]> {
    let summary = "Yield the result tensor out of a seki.fused_computation region";
    let arguments = (ins AnyTensor:$value);
    let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // SEKI_OPS_TD
