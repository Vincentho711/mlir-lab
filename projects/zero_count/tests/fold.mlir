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

func.func @test_no_fold_count_zeros_in_range_runtime_input(%x : i32) -> i32 {
    // CHECK: zero_count.count_zeros_in_range %arg0 {hi = 26 : i32, lo = 15 : i32} : i32
    %r = zero_count.count_zeros_in_range %x {lo = 15 : i32, hi = 26 : i32} : i32
    func.return %r : i32
}
