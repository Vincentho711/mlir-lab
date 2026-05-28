#ifndef ZERO_COUNT_OPS_TD
#define ZERO_COUNT_OPS_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===/

def ZeroCount_Dialect : Dialect {
    let name = "zero_count";
    let summary = "A dialect for counting zeros.";
    let cppNamespace = "::mlir::zero_count"; // the C++ namespace for the generated code
    let useDefaultAttributePrinterParser = 0;
    // fold() returns an IntegerAttr, MLIR needs to materialise it as a concerte op (an arith.constant).
    // It does this by calling materializeConstant on the dialect.
    // This tells ODS to generate the declaration: Operation *materializeConstant(OpBuilder &, Attribute, Type, Location);
    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Op base class
//===----------------------------------------------------------------------===/
class ZeroCount_Op<string mnemonic, list<Trait> traits = []>
    : Op<ZeroCount_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===/

def CountZerosOp : ZeroCount_Op<"count_zeros"> {
    let summary = "Count the numer of zero bits in a 32-bit integer.";
    let description = [{
        Returns the number of bits that are zero in the 32-bit input.

        Example:
            %result = zero_count.count_zeros %x : i32
            // if %x = 5 (binary: 00000000_00000000_00000000_00000101)
            // then %result = 29  (32 total bits, 3 are one, 29 are zero)
    }];

    let arguments = (ins I32:$input);
    let results   = (outs I32:$result);

    let assemblyFormat = "$input attr-dict `:` type($input)";

    let hasFolder = 1;
}

def CountZerosInRangeOp : ZeroCount_Op<"count_zeros_in_range"> {
    let summary = "Count zero bits in the half-open bit range [lo, hi) of an i32.";
    let description = [{
        Returns the number of zero bits in positions [lo, hi) of the input.
        Both lo and hi are compile-time interger attributes.

        Invariants verified at compile time:
            - lo >= 0
            - hi <= 32
            - lo < hi
        Example:
            // count zeros in the middle 16 bits (positions 8..23)
            %r = zero_count.count_zeros_in_range %x {lo = 8 : i32, hi = 24 : i32} : i32
    }];

    let arguments = (ins
        I32:$input,
        I32Attr:$lo,
        I32Attr:$hi
    );

    let results = (outs I32:$result);

    let assemblyFormat = "$input attr-dict `:` type($input)";

    let hasVerifier = 1;

    let hasFolder = 1;

    let hasCanonicalizer = 1;
}

def ClampOp : ZeroCount_Op<"clamp"> {
    let summary = "Clamp an i32 value to the compile-time range [lo, hi].";
    let description = [{
        Returns the input clamped to the closed range [lo, hi]:
            result = max(lo, min(hi, input))

        Both lo and hi are compile-time integer attributes.

        Folds:
            - lo = INT32_MIN, hi = INT32_MMAX : identity, returns input unchanged
            - lo = hi : returns constant lo
            - constant input : computes result at compile-time

        Example:
            %r = zero_count.clamp %x {lo = 0 : i32, hi = 255 : i32} : i32
    }];

    let arguments = (ins
        I32:$input,
        I32Attr:$lo,
        I32Attr:$hi
    );

    let results = (outs I32:$result);

    let assemblyFormat = "$input attr-dict `:` type($input)";

    let hasVerifier = 1;

    let hasFolder = 1;
}

def MatmulOp : ZeroCount_Op<"matmul"> {
    let summary = "Matrix multiply two 2-D f32 tensors.";

    let arguments = (ins
        AnyTensor:$lhs,
        AnyTensor:$rhs
    );

    let results = (outs AnyTensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";

    let hasVerifier = 1;
}

def BufMatmulOp : ZeroCount_Op<"buf_matmul"> {
    let summary = "Bufferised matmul on memrefs.";

    let arguments = (ins
        AnyMemRef:$lhs,
        AnyMemRef:$rhs,
        AnyMemRef:$result
    );

    let results = (outs);

    let assemblyFormat = "$lhs `,` $rhs `,` $result attr-dict `:` type($lhs) `,` type($rhs) `,` type($result)";

    let hasVerifier = 1;
}

#endif // ZERO_COUNT_OPS_TD
