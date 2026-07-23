// RUN: seki-opt --seki-attach-target="seki-target=seki-v1" \
// RUN:          --seki-cast-to-target-types %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_input_f32_to_bf16_cast
// CHECK: seki.cast {{.*}} : tensor<256x512xf32> -> tensor<256x512xbf16>
// CHECK: seki.cast {{.*}} : tensor<512x128xf32> -> tensor<512x128xbf16>
// CHECK: seki.matmul ins({{.*}} : tensor<256x512xbf16>, tensor<512x128xbf16>)
// CHECK-SAME:        outs({{.*}} : tensor<256x128xf32>)
func.func @matmul_input_f32_to_bf16_cast(%A : tensor<256x512xf32>, %B : tensor<512x128xf32>, 
                                      %C : tensor<256x128xf32>) -> tensor<256x128xf32>{
    %result = seki.matmul ins(%A, %B : tensor<256x512xf32>, tensor<512x128xf32>)
                outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
    return %result : tensor<256x128xf32>
}

// CHECK-LABEL: func.func @batch_matmul_input_f32_to_bf16_cast
// CHECK: seki.cast {{.*}} : tensor<4x256x512xf32> -> tensor<4x256x512xbf16>
// CHECK: seki.cast {{.*}} : tensor<4x512x128xf32> -> tensor<4x512x128xbf16>
// CHECK: seki.batch_matmul ins({{.*}} : tensor<4x256x512xbf16>, tensor<4x512x128xbf16>)
// CHECK-SAME:        outs({{.*}} : tensor<4x256x128xf32>)
func.func @batch_matmul_input_f32_to_bf16_cast(%A : tensor<4x256x512xf32>, %B : tensor<4x512x128xf32>, 
                                      %C : tensor<4x256x128xf32>) -> tensor<4x256x128xf32>{
    %result = seki.batch_matmul ins(%A, %B : tensor<4x256x512xf32>, tensor<4x512x128xf32>)
                outs(%C : tensor<4x256x128xf32>) -> tensor<4x256x128xf32>
    return %result : tensor<4x256x128xf32>
}

// CHECK-LABEL: func.func @bf16_no_cast
// CHECK-NOT: seki.cast {{.*}}
// CHECK: seki.matmul ins({{.*}} : tensor<256x512xbf16>, tensor<512x128xbf16>)
// CHECK-SAME:        outs({{.*}} : tensor<256x128xf32>)
func.func @bf16_no_cast(%A : tensor<256x512xbf16>, %B : tensor<512x128xbf16>, 
                        %C : tensor<256x128xf32>) -> tensor<256x128xf32>{
    %result = seki.matmul ins(%A, %B : tensor<256x512xbf16>, tensor<512x128xbf16>)
                outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
    return %result : tensor<256x128xf32>
}

// CHECK-LABEL: func.func @ew_binary_not_cast
// CHECK-NOT: seki.cast
// CHECK: seki.ew_binary< add> {{.*}} : tensor<256x128xf32>
func.func @ew_binary_not_cast(%lhs : tensor<256x128xf32>, %rhs : tensor<256x128xf32>) -> tensor<256x128xf32> {
    %r = seki.ew_binary<add> %lhs, %rhs : tensor<256x128xf32>
    return %r : tensor<256x128xf32>
}
