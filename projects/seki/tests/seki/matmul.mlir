// RUN: seki-opt %s | FileCheck %s


// CHECK-LABEL: func.func @linalg_matmul
func.func @linalg_matmul(%A : memref<4x8xf32>, %B : memref<8x3xf32>, %C : memref<4x3xf32>) {
    // CHECK: linalg.matmul
    // CHECK-SAME: ins(%{{.*}}, %{{.*}} : memref<4x8xf32>, memref<8x3xf32>)
    // CHECK-SAME: outs(%{{.*}} : memref<4x3xf32>)
    linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x3xf32>)
                  outs(%C : memref<4x3xf32>)
    func.return
}

// CHECK-LABEL: func.func @matmul_tensor
func.func @matmul_tensor(%A : tensor<4x8xf32>, %B : tensor<8x3xf32>) -> tensor<4x3xf32> {
    %zero = arith.constant 0.0 : f32

    // CHECK: tensor.empty()
    %C_empty = tensor.empty() : tensor<4x3xf32>

    // Returns a new tensor
    // CHECK: linalg.fill
    %C_zero = linalg.fill ins(%zero : f32) outs(%C_empty : tensor<4x3xf32>) -> tensor<4x3xf32>

    // CHECK: linalg.matmul
    %C = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x3xf32>) outs(%C_zero : tensor<4x3xf32>) -> tensor<4x3xf32>

    func.return %C : tensor<4x3xf32>
}
