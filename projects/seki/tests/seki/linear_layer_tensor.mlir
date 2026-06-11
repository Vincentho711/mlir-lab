// RUN: seki-opt %s | FileCheck %s

// Tensor-form layer : Y = ReLU(X@W + b)
// No memory address, pure value semantics

// CHECK-LABEL: func.func @linear_layer_tensor
func.func @linear_layer_tensor(%X : tensor<4x8xf32>, %W : tensor<8x3xf32>, %b : tensor<3xf32>) -> tensor<4x3xf32> {
    %zero = arith.constant 0.0 : f32

    %Y_empty = tensor.empty() : tensor<4x3xf32>
    %Y_zero = linalg.fill ins(%zero : f32) outs(%Y_empty : tensor<4x3xf32>) -> tensor<4x3xf32>
    %Y_mm = linalg.matmul ins(%X, %W : tensor<4x8xf32>, tensor<8x3xf32>) outs(%Y_zero : tensor<4x3xf32>) -> tensor<4x3xf32>

    // Bias add: Y[i, j] + b[j]
    %Y_bias = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d1)>,      // first ins -> %b
            affine_map<(d0, d1) -> (d0, d1)>   // first outs -> %Y_mm
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%b : tensor<3xf32>) outs(%Y_mm : tensor<4x3xf32>) {
    ^bb0(%b_val : f32, %y_val : f32):
        %sum = arith.addf %y_val, %b_val : f32
        linalg.yield %sum : f32
    } -> tensor<4x3xf32>

    // ReLU activation, in-place elementwise
    %Y_relu = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)> // first outs -> %Y_boas
        ],
        iterator_types = ["parallel", "parallel"] // describes the loop domain, ReLU iterates over a 2D output tensor<4x3xf32>, so there are 2 loop variable (d0, d1)
    } outs(%Y_bias : tensor<4x3xf32>)  {
    ^bb0(%y_val : f32) :
        %relu = arith.maximumf %y_val, %zero : f32
        linalg.yield %relu : f32
    } -> tensor<4x3xf32>

    func.return %Y_relu : tensor<4x3xf32>
}
