// RUN: seki-opt --seki-lower-to-seki-hw %s | FileCheck %s

// Verify that seki.relu lowers to seki_hw.tile_ew_unary<relu> and that the function signature is coverted from tensor to !seki_hw.tile<T>
// CHECK-LABEL: func.func @relu_lowering
// CHECK-SAME:    (%{{.*}}: !seki_hw.tile<f32>) -> !seki_hw.tile<f32>
// CHECK:         seki_hw.tile_ew_unary< relu>
// CHECK-NOT:     seki.relu
func.func @relu_lowering(%x: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.relu %x : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}
