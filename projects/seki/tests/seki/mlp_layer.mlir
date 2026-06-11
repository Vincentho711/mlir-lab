// RUN: seki-opt %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP
// RUN: seki-opt --verify-each %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

// CHECK-ROUNDTRIP-LABEL: func.func @valid_matmul_f32
// CHECK-ROUNDTRIP: seki.matmul ins(%{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<8x16xf32>)
// CHECK-ROUNDTRIP-SAME: outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
func.func @valid_matmul_f32(%A : tensor<4x8xf32>, %B : tensor<8x16xf32>, %C : tensor<4x16xf32>) -> tensor<4x16xf32> {
    %Y = seki.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf32>)
               outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %Y : tensor<4x16xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @mlp_layer_f32
// CHECK-ROUNDTRIP: seki.matmul ins(%{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<8x16xf32>)
// CHECK-ROUNDTRIP: seki.relu
func.func @mlp_layer_f32(%A : tensor<4x8xf32>, %B : tensor<8x16xf32>, %C : tensor<4x16xf32>) -> tensor<4x16xf32> {
    %Y_mm = seki.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf32>)
               outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
    %Y_relu = seki.relu %Y_mm : tensor<4x16xf32>
    return %Y_relu : tensor<4x16xf32>
}
