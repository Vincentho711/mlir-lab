// RUN: seki-opt --seki-attach-target="seki-target=seki-v1" --seki-tile-ops %s | FileCheck %s

// Test 1: standard mixed precision linear layer -- both A and B are cast f32 -> bf16
// before the matmul. Both standalone casts must be commuted: no full-tensor cast survives
// outside the lop.
// CHECK-LABEL: func.func @matmul_both_casted
// CHECK-NOT: seki.cast
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract_slice %arg0[{{.*}}] [128, 128] [1, 1] : tensor<256x512xf32> to tensor<128x128xf32>
// CHECK-NEXT: seki.cast {{.*}} : tensor<128x128xf32> -> tensor<128x128xbf16>
// CHECK: tensor.extract_slice %arg1[{{.*}}] [128, 256] [1, 1] : tensor<512x256xf32> to tensor<128x256xf32>
// CHECK-NEXT: seki.cast {{.*}} : tensor<128x256xf32> -> tensor<128x256xbf16>
// CHECK: seki.matmul ins({{.*}} : tensor<128x128xbf16>, tensor<128x256xbf16>)
func.func @matmul_both_casted(%A: tensor<256x512xf32>, %B: tensor<512x256xf32>,
                              %C: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %A_bf16 = seki.cast %A : tensor<256x512xf32> -> tensor<256x512xbf16>
    %B_bf16 = seki.cast %B : tensor<512x256xf32> -> tensor<512x256xbf16>
    %r = seki.matmul ins(%A_bf16, %B_bf16 : tensor<256x512xbf16>, tensor<512x256xbf16>)
                     outs(%C : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %r : tensor<256x256xf32>
}

// -----

// Test 2: batched attention-style Q @ K^T -- seki.batch_matmul with Q cast f32 -> bf16. Exercises rank reduction
// interacting correctly with the commutation.
// CHECK-LABEL: func.func @batch_matmul_q_cast
// CHECK-NOT: seki.cast
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<2x256x512xf32> to tensor<128x128xf32>
// CHECK-NEXT: seki.cast {{.*}} : tensor<128x128xf32> -> tensor<128x128xbf16>
// CHECK: seki.matmul ins({{.*}} : tensor<128x128xbf16>, tensor<128x256xbf16>)
// CHECK-SAME: outs({{.*}} : tensor<128x256xf32>) -> tensor<128x256xf32>
func.func @batch_matmul_q_cast(%Q: tensor<2x256x512xf32>, %K: tensor<2x512x256xbf16>,
                               %C: tensor<2x256x256xf32>) -> tensor<2x256x256xf32> {
    %Q_bf16 = seki.cast %Q : tensor<2x256x512xf32> -> tensor<2x256x512xbf16>
    %r = seki.batch_matmul ins(%Q_bf16, %K : tensor<2x256x512xbf16>, tensor<2x512x256xbf16>)
                           outs(%C : tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
    return %r : tensor<2x256x256xf32>
}

// -----
// Test 3: the cast feeds seki.ew_unary<relu>, not the tiled matmul directly; the second matmul's extract_slice sources
// the relu result, one hop away from the cast. Pattern only matches an immediate cast-of-source.
// CHECK-LABEL: func.func @cast_before_relu_not_commuted
// CHECK: scf.for
// CHECK: scf.for
// CHECK: seki.cast {{.*}} : tensor<256x256xf32> -> tensor<256x256xbf16>
// CHECK: seki.ew_unary< relu> {{.*}} : tensor<256x256xbf16>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract_slice {{.*}} : tensor<256x256xbf16> to tensor<128x128xbf16>
// CHECK-NOT: seki.cast {{.*}} tensor<128x128xf32>
// CHECK: seki.matmul
func.func @cast_before_relu_not_commuted(%A: tensor<256x256xbf16>, %B: tensor<256x256xbf16>,
                                         %Cacc: tensor<256x256xf32>, %D: tensor<256x256xbf16>,
                                         %Eacc: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %t = seki.matmul ins(%A, %B : tensor<256x256xbf16>, tensor<256x256xbf16>)
                     outs(%Cacc : tensor<256x256xf32>) -> tensor<256x256xf32>
    %t_bf16 = seki.cast %t : tensor<256x256xf32> -> tensor<256x256xbf16>
    %act = seki.ew_unary<relu> %t_bf16 : tensor<256x256xbf16>
    %out = seki.matmul ins(%act, %D : tensor<256x256xbf16>, tensor<256x256xbf16>)
                       outs(%Eacc : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %out : tensor<256x256xf32>
}
