// RUN: zero-count-opt %s | FileCheck %s

// CHECK-LABEL: func.func @tensor_matmul
func.func @tensor_matmul(%a : tensor<4x8xf32>, %b : tensor<8x3xf32>) -> tensor<4x3xf32> {
    // CHECK: zero_count.matmul
    // CHECK-SAME: (tensor<4x8xf32>, tensor<8x3xf32>) -> tensor<4x3xf32>
    %c = zero_count.matmul %a, %b : (tensor<4x8xf32>, tensor<8x3xf32>) -> tensor<4x3xf32>
    func.return %c : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @buf_matmul_default_memspace
func.func @buf_matmul_default_memspace(%a : memref<4x8xf32>, %b : memref<8x3xf32>, %c : memref<4x3xf32>) {
    // CHECK: zero_count.buf_matmul
    // CHECK-SAME: memref<4x8xf32>, memref<8x3xf32>, memref<4x3xf32>
    zero_count.buf_matmul %a, %b, %c : memref<4x8xf32>, memref<8x3xf32>, memref<4x3xf32>
    func.return
}

// CHECK-LABEL: func.func @buf_matmul_scratchpad
func.func @buf_matmul_scratchpad(%a : memref<4x8xf32, 1>, %b : memref<8x3xf32, 1>, %c : memref<4x3xf32, 1>) {
    // CHECK: zero_count.buf_matmul
    // CHECK-SAME: memref<4x8xf32, 1>, memref<8x3xf32, 1>, memref<4x3xf32, 1>
    zero_count.buf_matmul %a, %b, %c : memref<4x8xf32, 1>, memref<8x3xf32, 1>, memref<4x3xf32, 1>
    func.return
}
