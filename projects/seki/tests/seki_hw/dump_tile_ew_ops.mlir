// RUN: seki-opt --seki-hw-dump-tile-ew-ops %s 2>&1 | FileCheck %s

// Remarks appear before the module dump. Checks are  in walk order.

// @ew_dispatch
// CHECK: remark: seki_hw.tile_ew_unary: fn = {{.*}}relu
// CHECK: remark: seki_hw.tile_ew_binary: fn = {{.*}}add

// @all_unary_fns
// CHECK: remark: seki_hw.tile_ew_unary: fn = {{.*}}exp
// CHECK: remark: seki_hw.tile_ew_unary: fn = {{.*}}neg

// @non_ew_ops_skipped
// CHECK-NOT: remark: seki_hw.load
// CHECK-NOT: remark: seki_hw.tile_splat

func.func @ew_dispatch(
    %src : !seki_hw.tile<f32>,
    %rhs : !seki_hw.tile<f32>
) -> !seki_hw.tile<f32> {
    %relu = seki_hw.tile_ew_unary<relu> %src : !seki_hw.tile<f32>
    %add = seki_hw.tile_ew_binary<add> %relu, %rhs : !seki_hw.tile<f32>
    return %add : !seki_hw.tile<f32>
}

func.func @all_unary_fns(%t: !seki_hw.tile<f32>) -> !seki_hw.tile<f32> {
    %exp = seki_hw.tile_ew_unary<exp> %t : !seki_hw.tile<f32>
    %neg = seki_hw.tile_ew_unary<neg> %exp : !seki_hw.tile<f32>
    return %neg : !seki_hw.tile<f32>
}

func.func @non_ew_ops_skipped(%addr: i64) -> !seki_hw.tile<f32> {
    %tile = seki_hw.load %addr -> !seki_hw.tile<f32>
    %scalar = arith.constant 0.0 : f32
    %splat = seki_hw.tile_splat %scalar : f32 -> !seki_hw.tile<f32>
    return %tile : !seki_hw.tile<f32>
}
