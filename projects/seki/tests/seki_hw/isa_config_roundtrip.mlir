// RUN: seki-opt %s | FileCheck %s

// Full seki-v1 config — all four units present.
// Exercises parse/print for all five cap attrs in one shot.
// CHECK-LABEL: @full_config
// CHECK: #seki_hw.isa_config<
// CHECK-SAME: mac_array = #seki_hw.mac_array<inputType = bf16, outputTypes = [f32, bf16]>
// CHECK-SAME: vec_engine = #seki_hw.vec_engine<computeTypes = [bf16, f32], castPairs = [(bf16, f32), (f32, bf16)]>
// CHECK-SAME: scalar_unit = #seki_hw.scalar_unit<supportedTypes = [i64]>
// CHECK-SAME: dma_engine = #seki_hw.dma_engine<loadCastPairs = [(bf16, f32)], storeCastPairs = [(f32, bf16)]>
module @full_config attributes {"seki_hw.config" = #seki_hw.isa_config<
    mac_array   = #seki_hw.mac_array<inputType = bf16, outputTypes = [f32, bf16]>,
    vec_engine  = #seki_hw.vec_engine<computeTypes = [bf16, f32], castPairs = [(bf16, f32), (f32, bf16)]>,
    scalar_unit = #seki_hw.scalar_unit<supportedTypes = [i64]>,
    dma_engine  = #seki_hw.dma_engine<loadCastPairs = [(bf16, f32)], storeCastPairs = [(f32, bf16)]>
>} {}

// Scalar-only target — mac_array and vec_engine absent.
// Verifies the 'absent' keyword round-trips correctly.
// CHECK-LABEL: @absent_units
// CHECK: mac_array = absent
// CHECK-SAME: vec_engine = absent
module @absent_units attributes {"seki_hw.config" = #seki_hw.isa_config<
    mac_array   = absent,
    vec_engine  = absent,
    scalar_unit = #seki_hw.scalar_unit<supportedTypes = [i64]>,
    dma_engine  = #seki_hw.dma_engine<loadCastPairs = [], storeCastPairs = []>
>} {}
