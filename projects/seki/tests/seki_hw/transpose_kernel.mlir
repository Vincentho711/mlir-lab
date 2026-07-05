// RUN: seki-opt %s | FileCheck %s
// 128x128 tile transpose using tile_extract_vec / vec_insert_tile.

// CHECK-LABEL: func.func @tile_transpose
func.func @tile_transpose(%src: !seki_hw.tile<f32>, %dst_init: !seki_hw.tile<f32>) -> !seki_hw.tile<f32> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index

    // CHECK: scf.for
    // CHECK: tile_extract_vec< rows>
    // CHECK: vec_insert_tile< cols>
    %result = scf.for %i = %c0 to %c128 step %c1
               iter_args(%dst = %dst_init) -> (!seki_hw.tile<f32>) {
                   %idx = arith.index_cast %i : index to i64
                   %row = seki_hw.tile_extract_vec<rows> %src, %idx : !seki_hw.tile<f32>, i64 -> !seki_hw.vec<f32>
                   %dst_new = seki_hw.vec_insert_tile<cols> %row, %idx outs(%dst : !seki_hw.tile<f32>) : !seki_hw.vec<f32>, i64 -> !seki_hw.tile<f32>
                   scf.yield %dst_new : !seki_hw.tile<f32>
               }
               return %result : !seki_hw.tile<f32>
}
