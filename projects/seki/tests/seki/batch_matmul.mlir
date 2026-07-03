// RUN: seki-opt --verify-diagnostics --split-input-file %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

// Without transpose_b: A[B,M,K] x B[B,K,N] -> C[B,M,N]

// CHECK-ROUNDTRIP-LABEL: func.func @batch_matmul_basic
// CHECK-ROUNDTRIP: seki.batch_matmul
// CHECK-ROUNDTRIP-SAME: ins(%{{.*}}, %{{.*}} : tensor<2x4x8xf32>, tensor<2x8x16xf32>)
// CHECK-ROUNDTRIP-SAME: outs(%{{.*}} : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
func.func @batch_matmul_basic(
    %A: tensor<2x4x8xf32>, %B: tensor<2x8x16xf32>, %C: tensor<2x4x16xf32>
) -> tensor<2x4x16xf32> {
    %out = seki.batch_matmul ins(%A, %B : tensor<2x4x8xf32>, tensor<2x8x16xf32>)
                             outs(%C : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
    return %out : tensor<2x4x16xf32>
}

// -----

// With transpose_b: A[B,M,K] x B[B,N,K] -> C[B,M,N]

// CHECK-ROUNDTRIP-LABEL: func.func @batch_matmul_transpose_b
// CHECK-ROUNDTRIP: seki.batch_matmul
// CHECK-ROUNDTRIP-SAME: ins(%{{.*}}, %{{.*}} : tensor<2x4x8xf32>, tensor<2x16x8xf32>)
// CHECK-ROUNDTRIP-SAME: outs(%{{.*}} : tensor<2x4x16xf32>) {transpose_b} -> tensor<2x4x16xf32>
func.func @batch_matmul_transpose_b(
    %A: tensor<2x4x8xf32>, %B: tensor<2x16x8xf32>, %C: tensor<2x4x16xf32>
) -> tensor<2x4x16xf32> {
    %out = seki.batch_matmul ins(%A, %B : tensor<2x4x8xf32>, tensor<2x16x8xf32>)
                             outs(%C : tensor<2x4x16xf32>) {transpose_b} -> tensor<2x4x16xf32>
    return %out : tensor<2x4x16xf32>
}

// -----

// Verifier: batch dimension mismatch

func.func @batch_matmul_batch_mismatch(
    %A: tensor<2x4x8xf32>, %B: tensor<3x8x16xf32>, %C: tensor<2x4x16xf32>
) -> tensor<2x4x16xf32> {
    // expected-error @+1 {{batch mismatch}}
    %out = seki.batch_matmul ins(%A, %B : tensor<2x4x8xf32>, tensor<3x8x16xf32>)
                             outs(%C : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
    return %out : tensor<2x4x16xf32>
}

// -----

// Verifier: inner dimension mismatch (no transpose_b)

func.func @batch_matmul_inner_mismatch(
    %A: tensor<2x4x8xf32>, %B: tensor<2x99x16xf32>, %C: tensor<2x4x16xf32>
) -> tensor<2x4x16xf32> {
    // expected-error @+1 {{inner dimension mismatch}}
    %out = seki.batch_matmul ins(%A, %B : tensor<2x4x8xf32>, tensor<2x99x16xf32>)
                             outs(%C : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
    return %out : tensor<2x4x16xf32>
}

// -----

// Verifier: inner dimension mismatch (with transpose_b)

func.func @batch_matmul_inner_mismatch_transposed(
    %A: tensor<2x4x8xf32>, %B: tensor<2x16x99xf32>, %C: tensor<2x4x16xf32>
) -> tensor<2x4x16xf32> {
    // expected-error @+1 {{inner dimension mismatch}}
    %out = seki.batch_matmul ins(%A, %B : tensor<2x4x8xf32>, tensor<2x16x99xf32>)
                             outs(%C : tensor<2x4x16xf32>) {transpose_b} -> tensor<2x4x16xf32>
    return %out : tensor<2x4x16xf32>
}
