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

// Transpose on A must not be folded.
// CHECK-LABEL: func.func @transpose_a_not_fused
// CHECK: seki.transpose
// CHECK: seki.matmul
// CHECK-NOT: {transpose_b}
func.func @transpose_a_not_fused(%Q: tensor<128x128xbf16>, %K: tensor<128x128xbf16>, 
                                %acc: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %Q_T = seki.transpose %Q permutation = [1, 0] : tensor<128x128xbf16> -> tensor<128x128xbf16>
    %r = seki.matmul ins(%Q_T, %K : tensor<128x128xbf16>, tensor<128x128xbf16>)
                     outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %r : tensor<128x128xf32>
}

// Transpose reuslt feeds 2 consumers, must not fold.
// CHECK-LABEL: func.func @transpose_multi_use_not_fused
// CHECK: seki.transpose
// CHECK: seki.matmul
// CHECK-NOT: {transpose_b}
func.func @transpose_multi_use_not_fused(%Q: tensor<128x128xbf16>, %K: tensor<128x128xbf16>, 
                                         %acc1: tensor<128x128xf32>, %acc2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %K_T = seki.transpose %K permutation = [1, 0] : tensor<128x128xbf16> -> tensor<128x128xbf16>
    // %K_T has 2 uses, the matmul and ew_binary
    %r1 = seki.matmul ins(%Q, %K_T: tensor<128x128xbf16>, tensor<128x128xbf16>)
                      outs(%acc1: tensor<128x128xf32>) -> tensor<128x128xf32>
    %r2 = seki.matmul ins(%K_T, %Q: tensor<128x128xbf16>, tensor<128x128xbf16>)
                      outs(%acc2: tensor<128x128xf32>) -> tensor<128x128xf32>
    %sum = seki.ew_binary<add> %r1, %r2 : tensor<128x128xf32>
    return %sum : tensor<128x128xf32>
}
