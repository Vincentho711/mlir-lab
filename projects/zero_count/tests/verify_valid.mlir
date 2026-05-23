 // RUN: zero-count-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_count_zeros_in_range
func.func @test_count_zeros_in_range(%x : i32) -> i32 {
    // CHECK: zero_count.count_zeros_in_range %arg0 {hi = 20 : i32, lo = 5 : i32} : i32
    %r = zero_count.count_zeros_in_range %x {lo = 5 : i32, hi = 20 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_boundary_values
func.func @test_boundary_values(%x : i32) -> i32 {
    // CHECK: zero_count.count_zeros_in_range %arg0 {hi = 32 : i32, lo = 0 : i32} : i32
    %r = zero_count.count_zeros_in_range %x {lo = 0 : i32, hi = 32 : i32} : i32
    func.return %r : i32
}
