// RUN: seki-opt --verify-diagnostics --split-input-file %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

// 2D matrix transpose: [M,N] -> [N,M] with permutation [1,0]

// CHECK-ROUNDTRIP-LABEL: func.func @transpose_2d
// CHECK-ROUNDTRIP: seki.transpose %{{.*}} permutation = [1, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
func.func @transpose_2d(%x: tensor<4x8xf32>) -> tensor<8x4xf32> {
    %out = seki.transpose %x permutation = [1, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
    return %out : tensor<8x4xf32>
}

// -----

// 3D: [B,N,K] -> [B,K,N] with permutation [0,2,1]
// Used for transposed B operand in batched matmul before fusion folds it into {transpose_b}

// CHECK-ROUNDTRIP-LABEL: func.func @transpose_3d
// CHECK-ROUNDTRIP: seki.transpose %{{.*}} permutation = [0, 2, 1] : tensor<2x16x8xf32> -> tensor<2x8x16xf32>
func.func @transpose_3d(%x: tensor<2x16x8xf32>) -> tensor<2x8x16xf32> {
    %out = seki.transpose %x permutation = [0, 2, 1] : tensor<2x16x8xf32> -> tensor<2x8x16xf32>
    return %out : tensor<2x8x16xf32>
}

// -----

// Verifier: permutation size (1) does not match input rank (2)

func.func @transpose_perm_size_mismatch(%x: tensor<4x8xf32>) -> tensor<8x4xf32> {
    // expected-error @+1 {{permutation size must equal input rank}}
    %out = seki.transpose %x permutation = [1] : tensor<4x8xf32> -> tensor<8x4xf32>
    return %out : tensor<8x4xf32>
}

// -----

// Verifier: output shape disagrees with permuted input shape

func.func @transpose_shape_mismatch(%x: tensor<4x8xf32>) -> tensor<99x4xf32> {
    // expected-error @+1 {{output dim 0 must be input dim 1}}
    %out = seki.transpose %x permutation = [1, 0] : tensor<4x8xf32> -> tensor<99x4xf32>
    return %out : tensor<99x4xf32>
}
