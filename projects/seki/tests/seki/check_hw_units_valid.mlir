// RUN: seki-opt --seki-attach-target="seki-target=seki-v1" \
// RUN:          --seki-check-hw-units %s | FileCheck %s

// seki-v1 : MAC array accepts bf16 inputs, f32 or bf16 output.
//           VEC engine accepts bf16 and f32.


// CHECK-LABEL: func.func @matmul_bf16_valid
// CHECK-NOT: seki.cast
// CHECK: seki.matmul ins({{.*}} : tensor<256x512xbf16>, tensor<512x128xbf16>)
// CHECK-SAME:        outs({{.*}} : tensor<256x128xf32>)
func.func @matmul_bf16_valid(%A: tensor<256x512xbf16>, %B: tensor<512x128xbf16>,
                             %C: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %r = seki.matmul ins(%A, %B : tensor<256x512xbf16>, tensor<512x128xbf16>)
                     outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
    return %r : tensor<256x128xf32>
}

// CHECK-LABEL: func.func @relu_bf16_ok
// CHECK:       seki.ew_unary< relu>
func.func @relu_bf16_ok(%x : tensor<256x128xbf16>) -> tensor<256x128xbf16> {
    %r = seki.ew_unary<relu> %x : tensor<256x128xbf16>
    return %r : tensor<256x128xbf16>
}

// CHECK-LABEL: func.func @add_bf16_ok
// CHECK: seki.ew_binary< add>
func.func @add_bf16_ok(%lhs: tensor<256x128xbf16>, %rhs : tensor<256x128xbf16>) -> tensor<256x128xbf16> {
    %r = seki.ew_binary<add> %lhs, %rhs : tensor<256x128xbf16>
    return %r : tensor<256x128xbf16>
}
