// RUN: seki-opt --seki-lift-to-seki --split-input-file %s | FileCheck %s

// Check linalg.matmul -> seki.matmul
// CHECK-LABEL: func.func @named_matmul
// CHECK: seki.matmul
// CHECK-NOT: linalg.matmul
func.func @named_matmul(%A: tensor<128x512xf32>, %B: tensor<512x256xf32>, %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<128x512xf32>, tensor<512x256xf32>)
                       outs(%C : tensor<128x256xf32>) -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
}

// Check linalg.generic with the right index map and loop structure (contraction) -> seki.matmul
// CHECK-LABEL: func.func @generic_matmul
// CHECK: seki.matmul
// CHECK-NOT: linalg.generic
func.func @generic_matmul(%A: tensor<128x512xf32>, %B: tensor<512x256xf32>, %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(m, k, n) -> (m,k)>,
                         affine_map<(m, k, n) -> (k,n)>,
                         affine_map<(m, k, n) -> (m,n)>],
        // k disappears from the output map, reduction
        iterator_types = ["parallel","reduction","parallel"]
    } ins(%A, %B : tensor<128x512xf32>, tensor<512x256xf32>)
      outs(%C: tensor<128x256xf32>) {
    // %c is the running accumulator, as k is a reduction iterator, C[m, n] is read and written on every k iteration
    // linalg.yield feeds the result back in as %c for the next k.
    ^bb0(%a: f32, %b: f32, %c: f32):
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %mul, %c : f32
        linalg.yield %add : f32
    } -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
}

// Check linalg.generic -> seki.ew_unary
// CHECK-LABEL: func.func @ew_relu
// CHECK: seki.ew_unary< relu>
// CHECK-NOT: linalg.generic
func.func @ew_relu(%input: tensor<400x567xf32>, %output: tensor<400x567xf32>) -> tensor<400x567xf32> {
    %0 = linalg.generic{
        indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<400x567xf32>)
      outs(%output : tensor<400x567xf32>) {
    ^bb0(%in: f32, %out: f32):
        %zero = arith.constant 0.0 : f32
        %result = arith.maximumf %in, %zero : f32
        linalg.yield %result : f32
    } -> tensor<400x567xf32>
    return %0 : tensor<400x567xf32>
}

// Check linalg.generic with 2 ins operands and addf body -> seki.ew_binary<add>
// CHECK-LABEL: func.func @ew_add
// CHECK: seki.ew_binary< add>
// CHECK-NOT: linalg.generic
func.func @ew_add(%lhs: tensor<400x567xf32>, %rhs: tensor<400x567xf32>,
                  %output: tensor<400x567xf32>) -> tensor<400x567xf32> {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<400x567xf32>, tensor<400x567xf32>)
      outs(%output : tensor<400x567xf32>) {
    ^bb0(%in_lhs: f32, %in_rhs: f32, %out: f32):
        %result = arith.addf %in_lhs, %in_rhs: f32
        linalg.yield %result : f32
    } -> tensor<400x567xf32>
    return %0 : tensor<400x567xf32>
}

// Check that linalg.generic {} through aggregating the elements by row or column -> seki.reduce<>
// CHECK-LABEL: func.func @reduce_sum
// CHECK: seki.reduce< sum>
// CHECK-NOT: linalg.generic
func.func @reduce_sum(%input: tensor<819x392xf32>, %output: tensor<819xf32>) -> tensor<819xf32> {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
    } ins(%input: tensor<819x392xf32>)
      outs(%output : tensor<819xf32>) {
    ^bb0(%in: f32, %acc: f32):
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
    } -> tensor<819xf32>
    return %0 : tensor<819xf32>
}

// Check that linalg.broadcast -> seki.broadcast
// CHECK-LABEL: func.func @broadcast
// CHECK:       seki.broadcast
// CHECK-SAME:  broadcast_dims = [0, 1]
// CHECK-NOT:   linalg.broadcast
func.func @broadcast(%input: tensor<512x512xf32>,
                     %output: tensor<4x8x512x512xf32>) -> tensor<4x8x512x512xf32> {
    %0 = linalg.broadcast ins(%input : tensor<512x512xf32>)
                          outs(%output : tensor<4x8x512x512xf32>)
                          dimensions = [0, 1]
    return %0 : tensor<4x8x512x512xf32>
}

// Check linalg.batch_matmul -> seki.batch_matmul
// CHECK-LABEL: func.func @batch_matmul
// CHECK: seki.batch_matmul
// CHECK-NOT: linalg.batch_matmul
func.func @batch_matmul(%A: tensor<4x128x512xf32>, %B: tensor<4x512x256xf32>, %C: tensor<4x128x256xf32>) ->
    tensor<4x128x256xf32> {
    %0 = linalg.batch_matmul ins(%A, %B: tensor<4x128x512xf32>, tensor<4x512x256xf32>)
                             outs(%C: tensor<4x128x256xf32>) -> tensor<4x128x256xf32>
    return %0 : tensor<4x128x256xf32>
}

// CHECK linalg.transpose -> seki.transpose
// CHECK-LABEL: func.func @transpose_2d
// CHECK: seki.transpose
// CHECK-SAME: permutation = [1, 0]
// CHECK-NOT: linalg.transpose
func.func @transpose_2d(%input: tensor<128x256xf32>, %output: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %0 = linalg.transpose ins(%input: tensor<128x256xf32>)
                          outs(%output: tensor<256x128xf32>)
                          permutation = [1, 0]
    return %0 : tensor<256x128xf32>
}

// Check tensor.collapse_shape -> seki.reshape
// CHECK-LABEL: func.func @reshape_collapse
// CHECK: seki.reshape
// CHECK-NOT: tensor.collapse_shape
func.func @reshape_collapse(%input: tensor<4x128x256xf32>) -> tensor<512x256xf32> {
    %0 = tensor.collapse_shape %input [[0, 1], [2]]
        : tensor<4x128x256xf32> into tensor<512x256xf32>
    return %0 : tensor<512x256xf32>
}

// Check linalg.generic with transposed-B -> seki.matmul {transpose_b}
// CHECK-LABEL: func.func @generic_matmul_transpose_b
// CHECK: seki.matmul
// CHECK-SAME: transpose_b
// CHECK-NOT: linalg.generic
func.func @generic_matmul_transpose_b(%A: tensor<128x512xf32>,
                                      %B: tensor<256x512xf32>,
                                      %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(m, k, n) -> (m, k)>,
                         affine_map<(m, k, n) -> (n, k)>,
                         affine_map<(m, k, n) -> (m, n)>],
        iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%A, %B : tensor<128x512xf32>, tensor<256x512xf32>)
      outs(%C: tensor<128x256xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %mul, %c : f32
        linalg.yield %add : f32
    } -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
}
