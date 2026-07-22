// End to end pipeline test that covers multiple passes from --seki-verify-static-shapes
// to --seki-check-hw-units
// two layer MLP Y = relu(X @W1) @ W2
//
// RUN: seki-opt \
// RUN:     --seki-verify-static-shapes \
// RUN:     --seki-lift-to-seki \
// RUN:     --seki-attach-target="seki-target=seki-v1" \
// RUN:     --seki-cast-to-target-types \
// RUN:     --seki-check-hw-units \
// RUN:     %s | FileCheck %s

// X[256x512] @ W1[512x128] -> hideen[256x128] -> relu -> hidden_act[256x128]
// hidden_act[256x128] @ W2[128x128] -> Y[256x128]

// CHECK-LABEL: func.func @mlp_two_layer

// Layer 1: narrows X and W1 from f32 to bf16
// CHECK: seki.cast {{.*}} : tensor<256x512xf32> -> tensor<256x512xbf16>
// CHECK: seki.cast {{.*}} : tensor<512x128xf32> -> tensor<512x128xbf16>
// CHECK: seki.matmul ins({{.*}} : tensor<256x512xbf16>, tensor<512x128xbf16>)
// CHECK-SAME:        outs({{.*}} : tensor<256x128xf32>)

// Relu between layers, linalg.generic to seki.ew_unary<relu>
// CHECK: seki.ew_unary< relu>

// Layer 2: Narrows the relu output (f32) and W2(f32) to bf16
// CHECK: seki.cast {{.*}} : tensor<256x128xf32> -> tensor<256x128xbf16>
// CHECK: seki.cast {{.*}} : tensor<128x128xf32> -> tensor<128x128xbf16>
// CHECK: seki.matmul ins({{.*}} : tensor<256x128xbf16>, tensor<128x128xbf16>)
// CHECK-SAME:        outs({{.*}} : tensor<256x128xf32>)

func.func @mlp_two_layer(
    %X : tensor<256x512xf32>,
    %W1 : tensor<512x128xf32>,
    %W2: tensor<128x128xf32>
) -> tensor<256x128xf32> {
    %zero_f32 = arith.constant 0.0 : f32

    // Layer 1 matmul accumulator, zero-initialised
    %acc1 = tensor.splat %zero_f32 : tensor<256x128xf32>
    %hidden = linalg.matmul
        ins(%X, %W1 : tensor<256x512xf32>, tensor<512x128xf32>)
        outs(%acc1 : tensor<256x128xf32>) -> tensor<256x128xf32>

    // Relu activation between layers
    %relu_init = tensor.empty() : tensor<256x128xf32>
    %hidden_act = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%hidden : tensor<256x128xf32>)
      outs(%relu_init : tensor<256x128xf32>) {
      ^bb0(%in : f32, %out : f32):
          %act = arith.maximumf %in, %zero_f32 : f32
          linalg.yield %act : f32
    } -> tensor<256x128xf32>

    // Layer 2 matmul, input is the f32 relu output
    %acc2 = tensor.splat %zero_f32 : tensor<256x128xf32>
    %Y = linalg.matmul
        ins(%hidden_act, %W2 : tensor<256x128xf32>, tensor<128x128xf32>)
        outs(%acc2 : tensor<256x128xf32>) -> tensor<256x128xf32>
    return %Y : tensor<256x128xf32>
}
