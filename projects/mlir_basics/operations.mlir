// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @add
// CHECK:         arith.addi
// CHECK-LABEL: func.func @use_constant
// CHECK:         arith.constant 52

module {
    // This function adds two 32-bit integers.
    func.func @add(%a: i32, %b: i32) -> i32 {
        %result = arith.addi %a, %b : i32
        func.return %result : i32
    }

    // This function returns a constant of 52
    func.func @use_constant() -> i32 {
        %c52 = arith.constant 52 : i32
        func.return %c52 : i32
    }
}
