// RUN: seki-opt --verify-diagnostics --split-input-file %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

// Flatten 2D -> 1D

// CHECK-ROUNDTRIP-LABEL: func.func @reshape_flatten
// CHECK-ROUNDTRIP: seki.reshape %{{.*}} : tensor<4x8xf32> -> tensor<32xf32>
func.func @reshape_flatten(%x: tensor<4x8xf32>) -> tensor<32xf32> {
    %out = seki.reshape %x : tensor<4x8xf32> -> tensor<32xf32>
    return %out : tensor<32xf32>
}

// -----

// Expand 1D -> 2D

// CHECK-ROUNDTRIP-LABEL: func.func @reshape_expand
// CHECK-ROUNDTRIP: seki.reshape %{{.*}} : tensor<32xf32> -> tensor<4x8xf32>
func.func @reshape_expand(%x: tensor<32xf32>) -> tensor<4x8xf32> {
    %out = seki.reshape %x : tensor<32xf32> -> tensor<4x8xf32>
    return %out : tensor<4x8xf32>
}

// -----

// Verifier: element count mismatch (32 != 33)

func.func @reshape_count_mismatch(%x: tensor<4x8xf32>) -> tensor<33xf32> {
    // expected-error @+1 {{element count mismatch: input=32 output=33}}
    %out = seki.reshape %x : tensor<4x8xf32> -> tensor<33xf32>
    return %out : tensor<33xf32>
}

// -----

// Verifier: element type mismatch (f32 vs i32)

func.func @reshape_type_mismatch(%x: tensor<4xf32>) -> tensor<4xi32> {
    // expected-error @+1 {{element type mismatch}}
    %out = seki.reshape %x : tensor<4xf32> -> tensor<4xi32>
    return %out : tensor<4xi32>
}
