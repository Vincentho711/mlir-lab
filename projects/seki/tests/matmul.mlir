// RUN: zero-count-opt %s | FileCheck %s


// CHECK-LABEL: func.func @linalg_matmul
func.func @linalg_matmul(%A : memref<4x8xf32>, %B : memref<8x3xf32>, %C : memref<4x3xf32>) {
    // CHECK: linalg.matmul
    // CHECK-SAME: ins(%{{.*}}, %{{.*}} : memref<4x8xf32>, memref<8x3xf32>)
    // CHECK-SAME: outs(%{{.*}} : memref<4x3xf32>)
    linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x3xf32>)
                  outs(%C : memref<4x3xf32>)
    func.return
}
