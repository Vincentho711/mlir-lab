// RUN: zero-count-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @test_fold_count_zeros
func.func @test_fold_count_zeros() -> i32 {
    // CHECK: arith.constant 24 : i32
    // CHECK-NOT: zero_count.count_zeros
    %c = arith.constant 255 : i32
    %r = zero_count.count_zeros %c : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_no_fold_count_zeros_runtime_input
func.func @test_no_fold_count_zeros_runtime_input(%x : i32) -> i32 {
    // CHECK: zero_count.count_zeros %arg0 : i32
    %r = zero_count.count_zeros %x : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_fold_count_zeros_in_range
func.func @test_fold_count_zeros_in_range() -> i32 {
    // CHECK: arith.constant 5 : i32
    // CHECK-NOT: zero_count.count_zeros_in_range
    %c = arith.constant 556 : i32
    %r = zero_count.count_zeros_in_range %c {lo = 1 : i32, hi = 10 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_no_fold_count_zeros_in_range_runtime_input
func.func @test_no_fold_count_zeros_in_range_runtime_input(%x : i32) -> i32 {
    // CHECK: zero_count.count_zeros_in_range %arg0 {hi = 26 : i32, lo = 15 : i32} : i32
    %r = zero_count.count_zeros_in_range %x {lo = 15 : i32, hi = 26 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_canonicalize_full_range
func.func @test_canonicalize_full_range(%x : i32) -> i32 {
    // CHECK: zero_count.count_zeros %arg0 : i32
    // CHECK-NOT: zero_count.count_zeros_in_range
    %r = zero_count.count_zeros_in_range %x {lo = 0 : i32, hi = 32 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_no_canonicalize_partial_range
func.func @test_no_canonicalize_partial_range(%x: i32) -> i32 {
    // CHECK: zero_count.count_zeros_in_range %arg0 {hi = 16 : i32, lo = 0 : i32} : i32
    %r = zero_count.count_zeros_in_range %x {lo = 0 : i32, hi = 16 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_clamp_identity
func.func @test_clamp_identity(%x : i32) -> i32 {
    // CHECK: return %arg0
    // CHECK-NOT: zero_count.clamp
    %r = zero_count.clamp %x {lo = -2147483648 : i32, hi = 2147483647 : i32} : i32
    func.return %r : i32
}

// Constant fold: lo == hi, result is always that constant
// CHECK-LABEL: func.func @test_clamp_degenerate
func.func @test_clamp_degenerate(%x: i32) -> i32 {
    // CHECK: arith.constant 42 : i32
    // CHECK-NOT: zero_count.clamp
    %r = zero_count.clamp %x {lo = 42 : i32, hi = 42 : i32} : i32
    func.return %r : i32
}

// Constant fold: constant input within range
// CHECK-LABEL: func.func @test_clamp_constant_within
func.func @test_clamp_constant_within() -> i32 {
    // CHECK: arith.constant 100 : i32
    // CHECK-NOT: zero_count.clamp
    %c = arith.constant 100 : i32
    %r = zero_count.clamp %c {lo = 0 : i32, hi = 255 : i32} : i32
    func.return %r : i32
}

// Constant fold: constant input below lo, clamped up
// CHECK-LABEL: func.func @test_clamp_constant_below
func.func @test_clamp_constant_below() -> i32 {
    // CHECK: arith.constant 0 : i32
    // CHECK-NOT: zero_count.clamp
    %c = arith.constant -5 : i32
    %r = zero_count.clamp %c {lo = 0 : i32, hi = 255 : i32} : i32
    func.return %r : i32
}

// Constant fold: constant input above hi, clamped down
// CHECK-LABEL: func.func @test_clamp_constant_above
func.func @test_clamp_constant_above() -> i32 {
    // CHECK: arith.constant 255 : i32
    // CHECK-NOT: zero_count.clamp
    %c = arith.constant 300 : i32
    %r = zero_count.clamp %c {lo = 0 : i32, hi = 255 : i32} : i32
    func.return %r : i32
}

// No fold: runtime input, partial range
// CHECK-LABEL: func.func @test_clamp_no_fold
func.func @test_clamp_no_fold(%x: i32) -> i32 {
    // CHECK: zero_count.clamp %arg0 {hi = 255 : i32, lo = 0 : i32} : i32
    %r = zero_count.clamp %x {lo = 0 : i32, hi = 255 : i32} : i32
    func.return %r : i32
}
