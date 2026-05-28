// RUN: zero-count-opt %s -split-input-file -verify-diagnostics

func.func @bad_lhs_rank(%a : tensor<4xf32>, %b : tensor<4x3xf32>) -> tensor<1x3xf32> {
    // expected-error @below {{lhs must be a rank-2 tensor, got rank 1}}
    %c = zero_count.matmul %a, %b : (tensor<4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    func.return %c : tensor<1x3xf32>
}

// -----

func.func @bad_k(%a : tensor<2x4xf32>, %b : tensor<5x3xf32>) -> tensor<2x3xf32> {
    // expected-error @below {{contracting dimension mismatch}}
    %c = zero_count.matmul %a, %b : (tensor<2x4xf32>, tensor<5x3xf32>) -> tensor<2x3xf32>
    func.return %c : tensor<2x3xf32>
}

// -----
func.func @bad_result_shape(%a : tensor <2x4xf32>, %b : tensor<4x3xf32>) -> tensor<9x9xf32> {
    // expected-error @below {{result shape must be}}
    %c = zero_count.matmul %a, %b : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<9x9xf32>
    func.return %c : tensor<9x9xf32>
}
