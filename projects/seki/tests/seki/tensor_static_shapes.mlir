// RUN: seki-opt --seki-verify-static-shapes %s | FileCheck %s

// CHECK-LABEL: func.func @tensor_static_matmul
func.func @tensor_static_matmul(%A: tensor<500x600xf32>, %B: tensor<600x400xf32>, %C: tensor<500x400xf32>) -> tensor<500x400xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<500x600xf32>, tensor<600x400xf32>)
                       outs(%C : tensor<500x400xf32>) -> tensor<500x400xf32>
    return %0 : tensor<500x400xf32>
}
