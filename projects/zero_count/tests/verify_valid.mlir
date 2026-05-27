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

// CHECK-LABEL: func.func @test_clamp_valid
func.func @test_clamp_valid(%x: i32) -> i32 {
    // CHECK: zero_count.clamp %arg0 {hi = 255 : i32, lo = 0 : i32} : i32
    %r = zero_count.clamp %x {lo = 0 : i32, hi = 255 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_clamp_identical_lo_and_hi
func.func @test_clamp_identical_lo_and_hi(%x: i32) -> i32 {
    // CHECK: zero_count.clamp %arg0 {hi = 13 : i32, lo = 13 : i32} : i32
    %r = zero_count.clamp %x {lo = 13 : i32, hi = 13 : i32} : i32
    func.return %r : i32
}
