// RUN: seki-opt --verify-diagnostics --split-input-file %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

// Roundtrip: all three reduction functions

// CHECK-ROUNDTRIP-LABEL: func.func @reduce_sum_axis0
// CHECK-ROUNDTRIP: seki.reduce< sum> %{{.*}} axis = 0 : tensor<3x4xf32> -> tensor<4xf32>
func.func @reduce_sum_axis0(%x: tensor<3x4xf32>) -> tensor<4xf32> {
    %out = seki.reduce<sum> %x axis = 0 : tensor<3x4xf32> -> tensor<4xf32>
    return %out : tensor<4xf32>
}

// -----

// CHECK-ROUNDTRIP-LABEL: func.func @reduce_max_axis1
// CHECK-ROUNDTRIP: seki.reduce< max> %{{.*}} axis = 1 : tensor<3x4xf32> -> tensor<3xf32>
func.func @reduce_max_axis1(%x: tensor<3x4xf32>) -> tensor<3xf32> {
    %out = seki.reduce<max> %x axis = 1 : tensor<3x4xf32> -> tensor<3xf32>
    return %out : tensor<3xf32>
}

// -----

// CHECK-ROUNDTRIP-LABEL: func.func @reduce_min_axis1
// CHECK-ROUNDTRIP: seki.reduce< min> %{{.*}} axis = 1 : tensor<3x4xf32> -> tensor<3xf32>
func.func @reduce_min_axis1(%x: tensor<3x4xf32>) -> tensor<3xf32> {
    %out = seki.reduce<min> %x axis = 1 : tensor<3x4xf32> -> tensor<3xf32>
    return %out : tensor<3xf32>
}

// -----

// Rank 3: axis in the middle

// CHECK-ROUNDTRIP-LABEL: func.func @reduce_rank3_axis1
// CHECK-ROUNDTRIP: seki.reduce< sum> %{{.*}} axis = 1 : tensor<2x8x4xf32> -> tensor<2x4xf32>
func.func @reduce_rank3_axis1(%x: tensor<2x8x4xf32>) -> tensor<2x4xf32> {
    %out = seki.reduce<sum> %x axis = 1 : tensor<2x8x4xf32> -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
}

// -----

// Verifier: axis out of range

func.func @reduce_axis_out_of_range(%x: tensor<3x4xf32>) -> tensor<3xf32> {
    // expected-error @+1 {{axis 2 out of range for rank-2 input}}
    %out = seki.reduce<sum> %x axis = 2 : tensor<3x4xf32> -> tensor<3xf32>
    return %out : tensor<3xf32>
}

// -----

// Verifier: output shape does not match input with axis removed

func.func @reduce_wrong_output_shape(%x: tensor<3x4xf32>) -> tensor<99xf32> {
    // expected-error @+1 {{output dim 0 (99) must match input dim 1 (4)}}
    %out = seki.reduce<sum> %x axis = 0 : tensor<3x4xf32> -> tensor<99xf32>
    return %out : tensor<99xf32>
}
