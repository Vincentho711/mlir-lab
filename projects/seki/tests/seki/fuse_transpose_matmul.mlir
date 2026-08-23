// RUN: seki-opt --seki-fuse-ops %s | FileCheck %s

// Test that seki.tranpose on B-operand folds into {transpose_b}.
// CHECK-LABEL: func.func @transpose_b_fused
// CHECK-NOT: seki.transpose
// CHECK: seki.matmul ins({{.*}} : tensor<128x128xbf16>, tensor<128x128xbf16>)
// CHECK-SAME: {transpose_b}
// CHECK: return
func.func @transpose_b_fused(%Q: tensor<128x128xbf16>, %K: tensor<128x128xbf16>,
                            %acc: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %K_T = seki.transpose %K permutation = [1, 0] : tensor<128x128xbf16> -> tensor<128x128xbf16>
    %r = seki.matmul ins(%Q, %K_T : tensor<128x128xbf16>, tensor<128x128xbf16>)
                     outs(%acc: tensor<128x128xf32>) -> tensor<128x128xf32>
    return %r : tensor<128x128xf32>
}

// -----
