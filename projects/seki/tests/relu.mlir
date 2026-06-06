// RUN: seki-opt %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP
// RUN: seki-opt --canonicalize %s | FileCheck %s --check-prefix=CHECK-DCE

// CHECK-ROUNDTRIP: func.func @relu_f32
// CHECK-ROUNDTRIP: seki.relu %{{.*}} : tensor<4xf32>
func.func @relu_f32(%input : tensor<4xf32>) -> tensor<4xf32> {
    %output = seki.relu %input : tensor<4xf32>
    return %output : tensor<4xf32>
}

// CHECK-DCE: func.func @dead_relu_f32
// CHECK-DCE-NOT: seki.relu %{{.*}} : tensor<4xf32>
func.func @dead_relu_f32(%input : tensor<4xf32>) -> tensor<4xf32> {
    %output = seki.relu %input : tensor<4xf32>
    return %input : tensor<4xf32>
}


