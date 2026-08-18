// RUN: seki-opt --seki-attach-target="seki-target=seki-v1" \
// RUN:          --seki-pad-to-tile-boundary %s | FileCheck %s
// RUN: seki-opt --seki-pad-to-tile-boundary %s | FileCheck %s --check-prefix=NO-TARGET

// Test 1: basic unaligned matmul
// CHECK-LABEL: func.func @matmul_unaligned
// CHECK-DAG:       } : tensor<100x500xbf16> to tensor<128x512xbf16>
// CHECK-DAG:       } : tensor<500x256xbf16> to tensor<512x256xbf16>
// CHECK-DAG:       } : tensor<100x256xf32> to tensor<128x256xf32>
// CHECK:       seki.matmul ins({{.*}} : tensor<128x512xbf16>, tensor<512x256xbf16>)
// CHECK-SAME:              outs({{.*}} : tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:       tensor.extract_slice {{.*}} : tensor<128x256xf32> to tensor<100x256xf32>
// CHECK:       return {{.*}} : tensor<100x256xf32>
// NO-TARGET-LABEL: func.func @matmul_unaligned
// NO-TARGET-NOT: tensor.pad
func.func @matmul_unaligned(%A : tensor<100x500xbf16>, %B : tensor<500x256xbf16>, 
                            %C : tensor<100x256xf32>) -> tensor<100x256xf32> {
    %r = seki.matmul ins(%A, %B : tensor<100x500xbf16>, tensor<500x256xbf16>)
                     outs(%C : tensor<100x256xf32>) -> tensor<100x256xf32>
    return %r : tensor<100x256xf32>
}

// Test 2: already tile-aligned, pass is a no-op
// typConverter.isLegal() returns true for every op: all dims are multiples of 128.
// CHECK-LABEL: func.func @matmul_aligned
// CHECK-NOT: tensor.pad
// CHECK: seki.matmul ins({{.*}} : tensor<128x256xbf16>, tensor<256x128xbf16>)
// CHECK-SAME: outs({{.*}} : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK-NOT: tensor.extract_slice
// CHECK: return {{.*}} : tensor<128x128xf32>
func.func @matmul_aligned(%A: tensor<128x256xbf16>, %B: tensor<256x128xbf16>,
                          %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %r = seki.matmul ins(%A, %B: tensor<128x256xbf16>, tensor<256x128xbf16>)
                     outs(%C: tensor<128x128xf32>) -> tensor<128x128xf32>
    return %r : tensor<128x128xf32>
}

// Test 3: matmul -> relu chain, extract_slice at return
// CHECK-LABEL: func.func @mlp_ew_propagation
// CHECK: tensor.pad
// CHECK: seki.matmul {{.*}} -> tensor<128x256xf32>
// CHECK: seki.ew_unary< relu> {{.*}} : tensor<128x256xf32>
// CHECK-NOT: tensor.extract_slice {{.*}} tensor<128x256xf32> to tensor<100x256xf32>
// CHECK: tensor.extract_slice {{.*}} : tensor<128x256xf32> to tensor<100x256xf32>
// CHECK: return {{.*}} : tensor<100x256xf32>
func.func @mlp_ew_propagation(%A: tensor<100x500xbf16>, %B: tensor<500x256xbf16>,
                              %C: tensor<100x256xf32>) -> tensor<100x256xf32> {
    %r = seki.matmul ins(%A, %B: tensor<100x500xbf16>, tensor<500x256xbf16>)
                     outs(%C: tensor<100x256xf32>) -> tensor<100x256xf32>
    %relu = seki.ew_unary<relu> %r: tensor<100x256xf32>
    return %relu : tensor<100x256xf32>
}

// Test 4: batch_matmul, batch dimension must not be padded
// CHECK-LABEL: func.func @batch_matmul_batch_dim_preserved
// CHECK-DAG:   } : tensor<4x100x500xbf16> to tensor<4x128x512xbf16>
// CHECK-DAG:   } : tensor<4x500x256xbf16> to tensor<4x512x256xbf16>
// CHECK-DAG:   } : tensor<4x100x256xf32> to tensor<4x128x256xf32>
// CHECK:       seki.batch_matmul ins({{.*}} : tensor<4x128x512xbf16>, tensor<4x512x256xbf16>)
// CHECK-SAME:                    outs({{.*}} : tensor<4x128x256xf32>) -> tensor<4x128x256xf32>
// CHECK:       tensor.extract_slice {{.*}} : tensor<4x128x256xf32> to tensor<4x100x256xf32>
// CHECK:       return {{.*}} : tensor<4x100x256xf32>
func.func @batch_matmul_batch_dim_preserved(%A: tensor<4x100x500xbf16>, %B: tensor<4x500x256xbf16>,
                                            %C: tensor<4x100x256xf32>) -> tensor<4x100x256xf32> {
    %r = seki.batch_matmul ins(%A, %B: tensor<4x100x500xbf16>, tensor<4x500x256xbf16>)
                           outs(%C: tensor<4x100x256xf32>) -> tensor<4x100x256xf32>
    return %r : tensor<4x100x256xf32>
}

//Test 5: no target attached, pass is unconditionally a no-op
// Without --seki-attach-target, the module carries no #seki.target<> attribute.
// hasMACArray() returns false and runOnOperation() returns immediately.
// The IR must be byte-for-byte identical to the input.

// NO-TARGET-LABEL: func.func @no_target_no_op
// NO-TARGET-NOT:   tensor.pad
// NO-TARGET:       seki.matmul ins({{.*}} : tensor<100x500xbf16>, tensor<500x256xbf16>)
// NO-TARGET-NOT:   tensor.extract_slice
func.func @no_target_no_op(%A: tensor<100x500xbf16>, %B: tensor<500x256xbf16>,
                            %C: tensor<100x256xf32>) -> tensor<100x256xf32> {
    %r = seki.matmul ins(%A, %B : tensor<100x500xbf16>, tensor<500x256xbf16>)
                     outs(%C : tensor<100x256xf32>) -> tensor<100x256xf32>
    return %r : tensor<100x256xf32>
}
