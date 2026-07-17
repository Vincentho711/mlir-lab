// RUN: seki-opt %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

// CHECK-ROUNDTRIP-LABEL: func.func @cast_bf16_to_f32
// CHECK-ROUNDTRIP: seki.cast
func.func @cast_bf16_to_f32(%bf16_tensor: tensor<3x79xbf16>) -> tensor<3x79xf32> {
    %out = seki.cast %bf16_tensor : tensor<3x79xbf16> -> tensor<3x79xf32>
    return %out : tensor<3x79xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @cast_f32_to_bf16
// CHECK-ROUNDTRIP: seki.cast
func.func @cast_f32_to_bf16(%f32_tensor: tensor<3x79xf32>) -> tensor<3x79xbf16> {
    %out = seki.cast %f32_tensor : tensor<3x79xf32> -> tensor<3x79xbf16>
    return %out : tensor<3x79xbf16>
}
