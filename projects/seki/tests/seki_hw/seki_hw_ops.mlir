// RUN: seki-opt %s | FileCheck %s

// CHECK-LABEL: func.func @load_store_roundtrip
func.func @load_store_roundtrip(%addr_in : i64, %addr_out : i64) {
    // CHECK: seki_hw.load
    // CHECK: seki_hw.store
    %tile = seki_hw.load %addr_in -> !seki_hw.tile<f32>
    seki_hw.store %tile, %addr_out : !seki_hw.tile<f32>
    return
}

// CHECK-LABEL: func.func @load_all_element_types
func.func @load_all_element_types(%a: i64) {
    %t0 = seki_hw.load %a -> !seki_hw.tile<f32>
    %t1 = seki_hw.load %a -> !seki_hw.tile<bf16>
    %t2 = seki_hw.load %a -> !seki_hw.tile<i8>
    %t3 = seki_hw.load %a -> !seki_hw.tile<i32>
    return
}
