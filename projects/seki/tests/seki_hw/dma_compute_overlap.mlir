// RUN: seki-opt %s | FileCheck %s
// Verify that seki-opt accepts IR where a DMA load for tile K+1 is issued
// before the matmul for tile K completes. Double-buffering.
// SSA use-def chain encodes the ordering constraint
//    - seki_hw.matmul for tile 0 depends on %a0 and %b0 (must wait for them)
//    - %a1 = seki_hw.load has no edge into matmul 0
//    - the hardware scoreboard enforces timing; no explicit barrier needed

// CHECK-LABEL: func.func @dma_compute_overlap
func.func @dma_compute_overlap(
    %addr_a0 : i64, %addr_b0 : i64,
    %addr_a1 : i64, %addr_b1 : i64,
    %zero : f32
) -> (!seki_hw.tile<f32>, !seki_hw.tile<f32>) {
    // Tile 0 loads
    // CHECK: seki_hw.load
    %a0 = seki_hw.load %addr_a0 -> !seki_hw.tile<f32>
    // CHECK: seki_hw.load
    %b0 = seki_hw.load %addr_b0 -> !seki_hw.tile<f32>

    // Tile 1 prefetch, hoisted before matmul 0
    // CHECK: seki_hw.load
    %a1 = seki_hw.load %addr_a1 -> !seki_hw.tile<f32>

    // Tile 0 compute
    // CHECK: seki_hw.tile_splat
    %acc0 = seki_hw.tile_splat %zero : f32 -> !seki_hw.tile<f32>
    // CHECK: seki_hw.matmul
    %out0 = seki_hw.matmul ins(%a0, %b0 : !seki_hw.tile<f32>, !seki_hw.tile<f32>)
                           outs(%acc0 : !seki_hw.tile<f32>) -> !seki_hw.tile<f32>

    // Tile 1 second load can overlap with tail end of matmul 0
    // CHECK: seki_hw.load
    %b1 = seki_hw.load %addr_b1 -> !seki_hw.tile<f32>

    // Tile 1 compute
    // CHECK: seki_hw.tile_splat
    %acc1 = seki_hw.tile_splat %zero : f32 -> !seki_hw.tile<f32>
    // CHECK: seki_hw.matmul
    %out1 = seki_hw.matmul ins(%a1, %b1 : !seki_hw.tile<f32>, !seki_hw.tile<f32>)
                           outs(%acc1 : !seki_hw.tile<f32>) -> !seki_hw.tile<f32>

    return %out0, %out1 : !seki_hw.tile<f32>, !seki_hw.tile<f32>
}
