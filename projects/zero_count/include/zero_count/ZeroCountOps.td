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
}

#endif // ZERO_COUNT_OPS_TD
