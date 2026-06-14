// RUN: seki-opt %s | FileCheck %s

// CHECK-LABEL: func.func @tile_types
func.func @tile_types(
    %a: !seki_hw.tile<f32>,
    %b: !seki_hw.tile<bf16>,
    %c: !seki_hw.tile<i8>,
    %d: !seki_hw.tile<i32>
) {
    // CHECK: !seki_hw.tile<f32>
    // CHECK: !seki_hw.tile<bf16>
    // CHECK: !seki_hw.tile<i8>
    // CHECK: !seki_hw.tile<i32>
    return
}

// CHECK-LABEL: func.func @vec_types
func.func @vec_types(
  %a: !seki_hw.vec<f32>,
  %b: !seki_hw.vec<bf16>
) {
    // CHECK: !seki_hw.vec<f32>
    // CHECK: !seki_hw.vec<bf16>
    return
}
