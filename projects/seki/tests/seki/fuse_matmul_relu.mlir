// RUN: seki-opt --seki-fuse-ops %s | FileCheck %s

// matmul + relu fuses into one seki.fused_computation
// CHECK-LABEL: func.func @matmul_relu_fuse
// CHECK-NOT: seki.matmul
// CHECK: seki.fused_computation(%{{.*}}, %{{.*}}, %{{.*}} : tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK-NEXT: ^bb0(%{{.*}}: tensor<128x128xbf16>, %{{.*}}: tensor<128x128xbf16>, %{{.*}}: tensor<128x128xf32>):
// CHECK-NEXT: %[[T:.*]] = seki.matmul ins(%{{.*}}, %{{.*}} : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK-NEXT: %[[R:.*]] = seki.ew_unary< relu> %[[T]] : tensor<128x128xf32>
// CHECK-NEXT: seki.yield %[[R]] : tensor<128x128xf32>
// CHECK: return
func.func @matmul_relu_fuse(%A: tensor<128x128xbf16>, %B: tensor<128x128xbf16>,
                            %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %t = seki.matmul ins(%A, %B : tensor<128x128xbf16>, tensor<128x128xbf16>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    %r = seki.ew_unary<relu> %t : tensor<128x128xf32>
    return %r : tensor<128x128xf32>
}

// -----

// Fusing here would silently drop one consumer's input to the matmul result.
// Both consumers are kept live via a returned sum so DCE cannot remove either
// one before the pattern runs.
// CHECK-LABEL: func.func @matmul_multi_use_not_fused
// CHECK-NOT: seki.fused_computation
// CHECK: seki.matmul
// CHECK: seki.ew_unary< relu>
// CHECK: seki.ew_unary< exp>
func.func @matmul_multi_use_not_fused(%A: tensor<128x128xbf16>, %B: tensor<128x128xbf16>,
                                      %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %t = seki.matmul ins(%A, %B : tensor<128x128xbf16>, tensor<128x128xbf16>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    %r1 = seki.ew_unary<relu> %t : tensor<128x128xf32>
    %r2 = seki.ew_unary<exp> %t : tensor<128x128xf32>
    %sum = seki.ew_binary<add> %r1, %r2 : tensor<128x128xf32>
    return %sum : tensor<128x128xf32>
}
