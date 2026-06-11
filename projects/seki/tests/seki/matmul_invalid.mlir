// RUN: seki-opt %s --verify-diagnostics

func.func @bad_inner_dim(
    %A : tensor<4x8xf32>,
    %B : tensor<9x16xf32>,
    %C : tensor<4x16xf32>
) -> tensor<4x16xf32> {
    // expected-error @+1 {{inner dimensions must match: A columns (8) != B rows (9)}}
    %out = seki.matmul ins(%A, %B : tensor<4x8xf32>, tensor<9x16xf32>)
                       outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %out : tensor<4x16xf32>
}

func.func @element_type_mismatch(
    %A : tensor<4x8xf32>,
    %B : tensor<8x16xf16>,
    %C : tensor<4x16xf32>
) -> tensor<4x16xf32> {
    // expected-error @+1 {{A and B element types must match}}
    %out = seki.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf16>)
                        outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %out : tensor<4x16xf32>
}

func.func @integer_type_mismatch(
    %A : tensor<4x8xi32>,
    %B : tensor<8x16xi32>,
    %C : tensor<4x16xi32>
) -> tensor<4x16xi32> {
    // expected-error @+1 {{A must have floating-point element type}}
    %out = seki.matmul ins(%A, %B : tensor<4x8xi32>, tensor<8x16xi32>)
                        outs(%C : tensor<4x16xi32>) -> tensor<4x16xi32>
    return %out : tensor<4x16xi32>
}
