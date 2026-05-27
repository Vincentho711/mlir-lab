// RUN: zero-count-opt --lower-zero-count %s | FileCheck %s
//
// Lowering test: verify that zero_count.count_zeros is fully replaced
// by arith + math ops and no zero_count ops remain in the output.

// CHECK-LABEL: func.func @test_lower_count_zeros
func.func @test_lower_count_zeros(%x: i32) -> i32 {
  // CHECK-NOT: zero_count.count_zeros
  // CHECK:     arith.constant -1 : i32
  // CHECK:     arith.xori
  // CHECK:     math.ctpop
  %result = zero_count.count_zeros %x : i32
  func.return %result : i32
}

// CHECK-LABEL: func.func @test_lower_count_zeros_in_range
func.func @test_lower_count_zeros_in_range(%x : i32) -> i32 {
    // CHECK-NOT: zero_count.count_zeros_in_range
    // CHECK: arith.constant
    // CHECK: arith.andi
    // CHECK: arith.xori
    // CHECK: math.ctpop
    %r = zero_count.count_zeros_in_range %x {lo = 4 : i32, hi = 8 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_lower_in_range_low_boundary
func.func @test_lower_in_range_low_boundary(%x: i32) -> i32 {
    // CHECK-NOT: zero_count.count_zeros_in_range
    // CHECK: arith.constant 255 : i32
    // CHECK: arith.andi
    // CHECK: arith.xori
    // CHECK: math.ctpop
    %r = zero_count.count_zeros_in_range %x {lo = 0 : i32, hi = 8 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_lower_in_range_high_boundary
func.func @test_lower_in_range_high_boundary(%x: i32) -> i32 {
    // CHECK-NOT: zero_count.count_zeros_in_range
    // CHECK: arith.constant -268435456 : i32
    // CHECK: arith.andi
    // CHECK: arith.xori
    // CHECK: math.ctpop
    %r = zero_count.count_zeros_in_range %x {lo = 28 : i32, hi = 32 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_lower_in_range_full
func.func @test_lower_in_range_full(%x: i32) -> i32 {
    // CHECK-NOT: zero_count.count_zeros_in_range
    // CHECK: arith.constant -1 : i32
    // CHECK-NOT: arith.andi
    // CHECK: arith.xori
    // CHECK: math.ctpop
    %r = zero_count.count_zeros_in_range %x {lo = 0 : i32, hi = 32 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_lower_in_range_single_bit
func.func @test_lower_in_range_single_bit(%x: i32) -> i32 {
    // CHECK-NOT: zero_count.count_zeros_in_range
    // CHECK: arith.constant 32 : i32
    // CHECK: arith.andi
    // CHECK: arith.xori
    // CHECK: math.ctpop
    %r = zero_count.count_zeros_in_range %x {lo = 5 : i32, hi = 6 : i32} : i32
    func.return %r : i32
}

// CHECK-LABEL: func.func @test_lower_clamp
func.func @test_lower_clamp(%x: i32) -> i32 {
    // CHECK-NOT: zero_count.clamp
    // CHECK:     arith.constant 0 : i32
    // CHECK:     arith.constant 255 : i32
    // CHECK:     arith.maxsi
    // CHECK:     arith.minsi
    %r = zero_count.clamp %x {lo = 0 : i32, hi = 255 : i32} : i32
    func.return %r : i32
}


