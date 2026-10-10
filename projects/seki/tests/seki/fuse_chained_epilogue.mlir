// RUN: seki-opt --seki-fuse-ops %s | FileCheck %s

// Shapes are deliberately varied across tests (128, 256, 384 multiples;
// square and rectangular) to demonstrate that EpilogueFusionPattern is
// tile-size-agnostic: it never inspects a concrete dimension, only moves
// Type objects around structurally. Tiling into fixed 128x128 chunks is a
// later pass (Phase 20/21) that hasn't run yet at this stage.

// Test 1: a 3-op chain absorbs into one group via repeated pattern firing.
// CHECK-LABEL: func.func @matmul_relu_exp_chain
// CHECK-NOT: seki.matmul
// CHECK: seki.fused_computation(%{{.*}}, %{{.*}}, %{{.*}} : tensor<256x256xbf16>, tensor<256x256xbf16>, tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK-NEXT: ^bb0(%{{.*}}: tensor<256x256xbf16>, %{{.*}}: tensor<256x256xbf16>, %{{.*}}: tensor<256x256xf32>):
// CHECK-NEXT: %[[T:.*]] = seki.matmul ins(%{{.*}}, %{{.*}} : tensor<256x256xbf16>, tensor<256x256xbf16>) outs(%{{.*}} : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK-NEXT: %[[R1:.*]] = seki.ew_unary< relu> %[[T]] : tensor<256x256xf32>
// CHECK-NEXT: %[[R2:.*]] = seki.ew_unary< exp> %[[R1]] : tensor<256x256xf32>
// CHECK-NEXT: seki.yield %[[R2]] : tensor<256x256xf32>
// CHECK: return
func.func @matmul_relu_exp_chain(%A: tensor<256x256xbf16>, %B: tensor<256x256xbf16>,
                                 %C: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %t = seki.matmul ins(%A, %B : tensor<256x256xbf16>, tensor<256x256xbf16>)
                     outs(%C : tensor<256x256xf32>) -> tensor<256x256xf32>
    %r1 = seki.ew_unary<relu> %t : tensor<256x256xf32>
    %r2 = seki.ew_unary<exp> %r1 : tensor<256x256xf32>
    return %r2 : tensor<256x256xf32>
}

// -----

// Test 2: bias-add alone. The broadcast's single-use result (%bias_b) is
// resolved as an "extra" operand of the add; its producer (seki.broadcast)
// is absorbed into the body, and its raw input (%bias, 1D) becomes the group's 4th block argument instead of the broadcast result.
// Rectangular shapes here: M=256, K=512, N=128.
// CHECK-LABEL: func.func @matmul_bias_add
// CHECK-NOT: seki.matmul
// CHECK-NOT: seki.broadcast
// CHECK: seki.fused_computation(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<256x512xbf16>, tensor<512x128xbf16>, tensor<256x128xf32>, tensor<128xf32>) -> tensor<256x128xf32>
// CHECK-NEXT: ^bb0(%{{.*}}: tensor<256x512xbf16>, %{{.*}}: tensor<512x128xbf16>, %{{.*}}: tensor<256x128xf32>, %{{.*}}: tensor<128xf32>):
// CHECK-NEXT: %[[T:.*]] = seki.matmul ins(%{{.*}}, %{{.*}} : tensor<256x512xbf16>, tensor<512x128xbf16>) outs(%{{.*}} : tensor<256x128xf32>) -> tensor<256x128xf32>
// CHECK-NEXT: %[[BB:.*]] = seki.broadcast %{{.*}} broadcast_dims = [1] : tensor<128xf32> -> tensor<256x128xf32>
// CHECK-NEXT: %[[ADD:.*]] = seki.ew_binary< add> %[[T]], %[[BB]] : tensor<256x128xf32>
// CHECK-NEXT: seki.yield %[[ADD]] : tensor<256x128xf32>
// CHECK: return
func.func @matmul_bias_add(%A: tensor<256x512xbf16>, %B: tensor<512x128xbf16>,
                           %C: tensor<256x128xf32>, %bias: tensor<128xf32>) -> tensor<256x128xf32> {
    %t = seki.matmul ins(%A, %B : tensor<256x512xbf16>, tensor<512x128xbf16>)
                     outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
    %bias_b = seki.broadcast %bias broadcast_dims = [1] : tensor<128xf32> -> tensor<256x128xf32>
    %biased = seki.ew_binary<add> %t, %bias_b : tensor<256x128xf32>
    return %biased : tensor<256x128xf32>
}

// -----

// Test 3: the realistic linear-layer pattern, bias-add immediately followed by an activation.
// First firing seeds {matmul, broadcast, add}; second firing absorbs relu into that
// same group. Shapes use a 384 (3x128) multiple to show non-power-of-2 tile counts work too.
// CHECK-LABEL: func.func @matmul_bias_relu
// CHECK-NOT: seki.matmul
// CHECK-NOT: seki.broadcast
// CHECK: seki.fused_computation(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<384x256xbf16>, tensor<256x256xbf16>, tensor<384x256xf32>, tensor<256xf32>) -> tensor<384x256xf32>
// CHECK-NEXT: ^bb0(%{{.*}}: tensor<384x256xbf16>, %{{.*}}: tensor<256x256xbf16>, %{{.*}}: tensor<384x256xf32>, %{{.*}}: tensor<256xf32>):
// CHECK-NEXT: %[[T:.*]] = seki.matmul ins(%{{.*}}, %{{.*}} : tensor<384x256xbf16>, tensor<256x256xbf16>) outs(%{{.*}} : tensor<384x256xf32>) -> tensor<384x256xf32>
// CHECK-NEXT: %[[BB:.*]] = seki.broadcast %{{.*}} broadcast_dims = [1] : tensor<256xf32> -> tensor<384x256xf32>
// CHECK-NEXT: %[[ADD:.*]] = seki.ew_binary< add> %[[T]], %[[BB]] : tensor<384x256xf32>
// CHECK-NEXT: %[[ACT:.*]] = seki.ew_unary< relu> %[[ADD]] : tensor<384x256xf32>
// CHECK-NEXT: seki.yield %[[ACT]] : tensor<384x256xf32>
// CHECK: return
func.func @matmul_bias_relu(%A: tensor<384x256xbf16>, %B: tensor<256x256xbf16>,
                            %C: tensor<384x256xf32>, %bias: tensor<256xf32>) -> tensor<384x256xf32> {
    %t = seki.matmul ins(%A, %B : tensor<384x256xbf16>, tensor<256x256xbf16>)
                     outs(%C : tensor<384x256xf32>) -> tensor<384x256xf32>
    %bias_b = seki.broadcast %bias broadcast_dims = [1] : tensor<256xf32> -> tensor<384x256xf32>
    %biased = seki.ew_binary<add> %t, %bias_b : tensor<384x256xf32>
    %act = seki.ew_unary<relu> %biased : tensor<384x256xf32>
    return %act : tensor<384x256xf32>
}

// -----

// Test 4: ambiguous guard. Both operands of the add are single-use matmul
// results — isChainOperand matches both, so the pattern bails rather than
// arbitrarily picking one. This is the deliberate scope boundary: general
// multi-input DAG fusion (SwiGLU-style) is deferred, not attempted here.
// CHECK-LABEL: func.func @ambiguous_chain_not_fused
// CHECK-NOT: seki.fused_computation
// CHECK: seki.matmul
// CHECK: seki.matmul
// CHECK: seki.ew_binary< add>
// CHECK: return
func.func @ambiguous_chain_not_fused(%A: tensor<256x256xbf16>, %B: tensor<256x256xbf16>,
                                     %C: tensor<256x256xbf16>, %D: tensor<256x256xbf16>,
                                     %E: tensor<256x256xf32>, %F: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %m1 = seki.matmul ins(%A, %B : tensor<256x256xbf16>, tensor<256x256xbf16>)
                      outs(%E : tensor<256x256xf32>) -> tensor<256x256xf32>
    %m2 = seki.matmul ins(%C, %D : tensor<256x256xbf16>, tensor<256x256xbf16>)
                      outs(%F : tensor<256x256xf32>) -> tensor<256x256xf32>
    %sum = seki.ew_binary<add> %m1, %m2 : tensor<256x256xf32>
    return %sum : tensor<256x256xf32>
}

// -----

// Test 5: seki.cast absorbs into the epilogue group, the mixed precision pattern (matmul -> relu -> cast -> next matmul).
// The cast narrows the f32 relu output to bf16 before feeding the next layer's matmul.
// The group's own result type (bf16) differs from its internal "chain" value's type (f32).
// Shapes match the manual 256x512/512x256/256x256 check run earlier.
// CHECK-LABEL: func.func @matmul_relu_cast_fuse
// CHECK-NOT: seki.matmul
// CHECK: seki.fused_computation(%{{.*}}, %{{.*}}, %{{.*}} : tensor<256x512xbf16>, tensor<512x256xbf16>, tensor<256x256xf32>) -> tensor<256x256xbf16>
// CHECK-NEXT: ^bb0(%{{.*}}: tensor<256x512xbf16>, %{{.*}}: tensor<512x256xbf16>, %{{.*}}: tensor<256x256xf32>):
// CHECK-NEXT: %[[T:.*]] = seki.matmul ins(%{{.*}}, %{{.*}} : tensor<256x512xbf16>, tensor<512x256xbf16>) outs(%{{.*}} : tensor<256x256xf32>) -> tensor<256x256xf32>
// CHECK-NEXT: %[[R:.*]] = seki.ew_unary< relu> %[[T]] : tensor<256x256xf32>
// CHECK-NEXT: %[[C:.*]] = seki.cast %[[R]] : tensor<256x256xf32> -> tensor<256x256xbf16>
// CHECK-NEXT: seki.yield %[[C]] : tensor<256x256xbf16>
// CHECK: return
func.func @matmul_relu_cast_fuse(%A: tensor<256x512xbf16>, %B: tensor<512x256xbf16>,
                                 %C: tensor<256x256xf32>) -> tensor<256x256xbf16> {
    %t = seki.matmul ins(%A, %B : tensor<256x512xbf16>, tensor<512x256xbf16>)
                     outs(%C: tensor<256x256xf32>) -> tensor<256x256xf32>
    %r = seki.ew_unary<relu> %t : tensor<256x256xf32>
    %cast = seki.cast %r : tensor<256x256xf32> -> tensor<256x256xbf16>
    return %cast : tensor<256x256xbf16>
}
