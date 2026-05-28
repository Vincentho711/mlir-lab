// RUN: zero-count-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_matmul_2x4_times_4x3
func.func @test_matmul_2x4_times_4x3(
    %a : tensor<2x4xf32>,
    %b : tensor<4x3xf32>) -> tensor<2x3xf32> {
    // CHECK: zero_count.matmul
    // CHECK-SAME: (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %c = zero_count.matmul %a, %b : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    func.return %c : tensor<2x3xf32>
}

func.func @test_matmul_square(
    %a : tensor<8x8xf32>,
    %b : tensor<8x8xf32>) -> tensor<8x8xf32> {
    // CHECK: zero_count.matmul
    // CHECK-SAME: (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    %c = zero_count.matmul %a, %b : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    func.return %c : tensor<8x8xf32>
}
