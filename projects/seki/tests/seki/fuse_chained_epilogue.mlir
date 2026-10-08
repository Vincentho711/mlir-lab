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
