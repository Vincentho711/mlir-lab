// RUN: seki-opt --seki-fuse-ops %s | FileCheck %s

// Test 1: a 3-op chain absorbs into one group via repeated pattern firing.
// CHECK-LABEL: func.func @matmul_relu_exp_chain
// CHECK-NOT: seki.matmul
// CHECK: seki.fused_computation(%{{.*}}, %{{.*}}, %{{.*}} : tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK-NEXT: ^bb0(%{{.*}}: tensor<128x128xbf16>, %{{.*}}: tensor<128x128xbf16>, %{{.*}}: tensor<128x128xf32>):
// CHECK-NEXT: %[[T:.*]] = seki.matmul ins(%{{.*}}, %{{.*}} : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%{{.*}} : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK-NEXT: %[[R1:.*]] = seki.ew_unary< relu> %[[T]] : tensor<128x128xf32>
// CHECK-NEXT: %[[R2:.*]] = seki.ew_unary< exp> %[[R1]] : tensor<128x128xf32>
// CHECK-NEXT: seki.yield %[[R2]] : tensor<128x128xf32>
// CHECK: return
func.func @matmul_relu_exp_chain(%A: tensor<128x128xbf16>, %B: tensor<128x128xbf16>,
                                 %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %t = seki.matmul ins(%A, %B : tensor<128x128xbf16>, tensor<128x128xbf16>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    %r1 = seki.ew_unary<relu> %t : tensor<128x128xf32>
    %r2 = seki.ew_unary<exp> %r1 : tensor<128x128xf32>
    return %r2 : tensor<128x128xf32>
}

// -----

// Test 2: bias-add alone. The broadcast's single-use result (%bias_b) is 
// resolved as an "extra" operand of the add; its producer (seki.broadcast)
// is absorted into the body, and its raw input (%bias, 1D) becomes the group's 4th block argument instead of the broadcast result.
// CHECK-LABEL: func.func @matmul_bias_add
// CHECK-NOT: seki.matmul
// CHECK-NOT: seki.broadcast
// CHECK: seki.fused_computation(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<128x128xf32>
// CHECK-NEXT: ^bb0(%{{.*}}: tensor<128x128xbf16>, %{{.*}}: tensor<128x128xbf16>, %{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128xf32>):
// CHECK-NEXT: %[[T:.*]] = seki.matmul ins(%{{.*}}, %{{.*}} : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%{{.*}} : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK-NEXT: %[[BB:.*]] = seki.broadcast %{{.*}} broadcast_dims = [1] : tensor<128xf32> -> tensor<128x128xf32>
// CHECK-NEXT: %[[ADD:.*]] = seki.ew_binary< add> %[[T]], %[[BB]] : tensor<128x128xf32>
// CHECK-NEXT: seki.yield %[[ADD]] : tensor<128x128xf32>
// CHECK: return
func.func @matmul_bias_add(%A: tensor<128x128xbf16>, %B: tensor<128x128xbf16>,
                           %C: tensor<128x128xf32>, %bias: tensor<128xf32>) -> tensor<128x128xf32> {
    %t = seki.matmul ins(%A, %B : tensor<128x128xbf16>, tensor<128x128xbf16>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    %bias_b = seki.broadcast %bias broadcast_dims = [1] : tensor<128xf32> -> tensor<128x128xf32>
    %biased = seki.ew_binary<add> %t, %bias_b : tensor<128x128xf32>
    return %biased : tensor<128x128xf32>
}

// -----

// Test 3: the realistic linear-layer patern, bias-add immediately followed by an activation.
// First firing seeds {matmul, broadcast, add}; second firing absorbs relu into that
// same group.
// CHECK-LABEL: func.func @matmul_bias_relu
// CHECk-NOT: seki.matmul
// CHECk-NOT: seki.broadcast
// CHECK: seki.fused_computation(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<128x128xf32>
// CHECK-NEXT: ^bb0(%{{.*}}: tensor<128x128xbf16>, %{{.*}}: tensor<128x128xbf16>, %{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128xf32>):
// CHECK-NEXT: %[[T:.*]] = seki.matmul ins(%{{.*}}, %{{.*}} : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%{{.*}} : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK-NEXT: %[[BB:.*]] = seki.broadcast %{{.*}} broadcast_dims = [1] : tensor<128xf32> -> tensor<128x128xf32>
// CHECK-NEXT: %[[ADD:.*]] = seki.ew_binary< add> %[[T]], %[[BB]] : tensor<128x128xf32>
// CHECK-NEXT: %[[ACT:.*]] = seki.ew_unary< relu> %[[ADD]] : tensor<128x128xf32>
// CHECK-NEXT: seki.yield %[[ACT]] : tensor<128x128xf32>
// CHECK: return
func.func @matmul_bias_relu(%A: tensor<128x128xbf16>, %B: tensor<128x128xbf16>,
                            %C: tensor<128x128xf32>, %bias: tensor<128xf32>) -> tensor<128x128xf32> {
    %t = seki.matmul ins(%A, %B : tensor<128x128xbf16>, tensor<128x128xbf16>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    %bias_b = seki.broadcast %bias broadcast_dims = [1] : tensor<128xf32> -> tensor<128x128xf32>
    %biased = seki.ew_binary<add> %t, %bias_b : tensor<128x128xf32>
    %act = seki.ew_unary<relu> %biased : tensor<128x128xf32>
    return %act : tensor<128x128xf32>
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
func.func @ambiguous_chain_not_fused(%A: tensor<128x128xbf16>, %B: tensor<128x128xbf16>,
                                     %C: tensor<128x128xbf16>, %D: tensor<128x128xbf16>,
                                     %E: tensor<128x128xf32>, %F: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %m1 = seki.matmul ins(%A, %B : tensor<128x128xbf16>, tensor<128x128xbf16>)
                      outs(%E : tensor<128x128xf32>) -> tensor<128x128xf32>
    %m2 = seki.matmul ins(%C, %D : tensor<128x128xbf16>, tensor<128x128xbf16>)
                      outs(%F : tensor<128x128xf32>) -> tensor<128x128xf32>
    %sum = seki.ew_binary<add> %m1, %m2 : tensor<128x128xf32>
    return %sum : tensor<128x128xf32>
}
