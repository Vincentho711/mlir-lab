// RUN: seki-opt  --seki-cast-to-target-types %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_no_target_noop
// CHECK-NOT: seki.cast
// CHECK: seki.matmul ins({{.*}} : tensor<256x512xf32>, tensor<512x128xf32>)
// CHECK-SAME:        outs({{.*}} : tensor<256x128xf32>)
func.func @matmul_no_target_noop(%A : tensor<256x512xf32>, %B : tensor<512x128xf32>, 
                                 %C : tensor<256x128xf32>) -> tensor<256x128xf32>{
    %result = seki.matmul ins(%A, %B : tensor<256x512xf32>, tensor<512x128xf32>)
                outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
    return %result : tensor<256x128xf32>
}
