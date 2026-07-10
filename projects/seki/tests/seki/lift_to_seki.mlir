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
