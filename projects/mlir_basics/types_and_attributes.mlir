// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @integer_types
// CHECK:         arith.extsi
// CHECK-LABEL: func.func @float_types
// CHECK:         arith.extf
// CHECK-LABEL: func.func @int_to_float
// CHECK:         arith.sitofp
// CHECK-LABEL: func.func @float_to_sint
// CHECK:         arith.fptosi
// CHECK-LABEL: func.func @comparisons
// CHECK:         arith.cmpi
// CHECK-LABEL: func.func @select_op
// CHECK:         arith.select
// CHECK-LABEL: func.func @truncation
// CHECK:         arith.trunci

module {
    // Integer types of different widths
    func.func @integer_types() -> i64 {
        %a = arith.constant 100 : i8
        %a_2 = arith.constant 100 : i8
        %b = arith.constant 200 : i32
        %c = arith.constant 300 : i64

        %a_to_i32 = arith.extsi %a : i8 to i32
        %sum_i32 = arith.addi %a_to_i32, %b : i32

        %sum_to_i64 = arith.extsi %sum_i32 : i32 to i64
        %result = arith.addi %sum_to_i64, %c : i64

        func.return %result : i64
    }

    // Float types and float operations
    func.func @float_types() -> f64 {
        %x = arith.constant 1.5 : f32
        %y = arith.constant 2.5 : f64

        %x_to_f64 = arith.extf %x : f32 to f64
        %result   = arith.addf %x_to_f64, %y : f64

        func.return %result : f64
    }

    // Converting between integers and floats
    func.func @int_to_float() -> f32 {
        %n = arith.constant 52 : i32
        %result = arith.sitofp %n : i32 to f32

        func.return %result : f32
    }

    // Converting between float and integers
    func.func @float_to_sint() -> i32 {
        %f = arith.constant 3.678 : f32
        %result = arith.fptosi %f : f32 to i32

        func.return %result : i32
    }

    // Index type
    func.func @index_type() -> index {
        %i = arith.constant 10 : index
        %j = arith.constant 20 : index
        %result = arith.addi %i, %j : index

        func.return %result : index
    }

    // Attributes vs SSA values
    func.func @attributes_vs_values() -> i32 {
        %c = arith.constant 42 : i32
        %t = arith.constant true
        %f = arith.constant false

        func.return %c : i32
    }

    // Comparison
    func.func @comparisons() -> i1 {
        %a  = arith.constant 10 : i32
        %b  = arith.constant 20 : i32

        // signed less than, lhs < rhs
        %less = arith.cmpi slt, %a, %b : i32
        // equal
        %equal = arith.cmpi eq, %a, %b : i32
        // signed greater than, lhs > rhs
        %greater = arith.cmpi sgt, %a, %b : i32
        // signed greater than or equal to, lhs >= rhs
        %greater_equal = arith.cmpi sge, %a, %b : i32

        func.return %greater_equal : i1
    }

    // Select operation
    // If the condition is true, return the first value, else the second.
    func.func @select_op() -> i32 {
        %a = arith.constant 40 : i32
        %b = arith.constant 41 : i32

        %cond = arith.cmpi slt, %a, %b : i32
        %result = arith.select %cond, %a, %b : i32

        func.return %result : i32
    }

    // Truncation, it throws away the high bits
    func.func @truncation() -> i8 {
        %n = arith.constant 300 : i32
        // 300 in binary : 0000_0001_0010_1100
        // i8 keeps only 8 bits : 0010_1100 = 44
        %result = arith.trunci %n : i32 to i8
        func.return %result : i8
    }

}
