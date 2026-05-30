// RUN: zero-count-opt %s | FileCheck %s

// CHECK-LABEL: func.func @linear_layer_matmul
func.func @linear_layer_matmul(%X : memref<4x8xf32>, %W : memref<8x3xf32>, %Y : memref<4x3xf32>) {
    %zero = arith.constant 0.0 : f32

    // CHECK: linalg.fill
    linalg.fill ins(%zero : f32) outs(%Y : memref<4x3xf32>)

    // CHECK: linalg.matmul
    linalg.matmul ins(%X, %W : memref<4x8xf32>, memref<8x3xf32>)
                  outs(%Y : memref<4x3xf32>)

    func.return
}

// CHECK-LABEL: func.func @linear_layer_with_bias
func.func @linear_layer_with_bias(%X : memref<4x8xf32>, %W : memref<8x3xf32>, %b : memref<3xf32>, %Y : memref<4x3xf32>) {
    %zero = arith.constant 0.0 : f32

    // CHECK: linalg.fill
    linalg.fill ins(%zero : f32) outs(%Y : memref<4x3xf32>)

    // CHECK: linalg.matmul
    linalg.matmul ins(%X, %W : memref<4x8xf32>, memref<8x3xf32>)
                  outs(%Y : memref<4x3xf32>)

    // Y[i,j] += b[j] - b broadcast across the batch dimension
    // CHECK: linalg.generic
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d1)>,     // b[j] - no batch dimension
            affine_map<(d0, d1) -> (d0, d1)>  // Y[i,j]
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%b : memref<3xf32>) outs(%Y : memref<4x3xf32>) {
    ^bb0(%b_val : f32, %y_val : f32):
        %sum = arith.addf %y_val, %b_val : f32
        linalg.yield %sum : f32
    }

    func.return
}

// CHECK-LABEL: func.func @linear_layer_with_bias_and_relu
func.func @linear_layer_with_bias_and_relu(%X : memref<4x8xf32>, %W : memref<8x3xf32>, %b : memref<3xf32>, %Y : memref<4x3xf32>) {
    %zero = arith.constant 0.0 : f32

    // CHECK: linalg.fill
    linalg.fill ins(%zero : f32) outs(%Y : memref<4x3xf32>)

    // CHECK: linalg.matmul
    linalg.matmul ins(%X, %W : memref<4x8xf32>, memref<8x3xf32>)
                  outs(%Y : memref<4x3xf32>)

    // Y[i,j] += b[j] - b broadcast across the batch dimension
    // CHECK: linalg.generic
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d1)>,     // b[j] - no batch dimension
            affine_map<(d0, d1) -> (d0, d1)>  // Y[i,j]
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%b : memref<3xf32>) outs(%Y : memref<4x3xf32>) {
    ^bb0(%b_val : f32, %y_val : f32):
        %sum = arith.addf %y_val, %b_val : f32
        linalg.yield %sum : f32
    }

    func.return
}
