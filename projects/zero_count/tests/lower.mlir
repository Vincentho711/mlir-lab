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
