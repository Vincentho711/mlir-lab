// RUN: seki-opt %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @scratchpad_linear_layer_tensor
func.func @scratchpad_linear_layer_tensor(
    %X : memref<4x8xf32>,
    %W : memref<8x3xf32>,
    %b : memref<3xf32>
) -> memref<4x3xf32> {
    %zero = arith.constant 0.000000e+00 : f32

    // CHECK: memref.alloc() {{.*}} : memref<4x3xf32, 1>
    %spad = memref.alloc() {alignment = 64 : i64} : memref<4x3xf32, 1>

    // CHECK: linalg.fill
    linalg.fill ins(%zero : f32) outs(%spad : memref<4x3xf32, 1>)
    // CHECK: linalg.matmul
    linalg.matmul ins(%X, %W : memref<4x8xf32>, memref<8x3xf32>) 
                  outs(%spad : memref<4x3xf32, 1>)

    // CHECK: linalg.generic
    linalg.generic {
        indexing_maps = [#map, #map1],
        iterator_types = ["parallel", "parallel"]
    } ins(%b : memref<3xf32>) outs(%spad : memref<4x3xf32, 1>) {
    ^bb0(%b_val: f32, %y_val: f32):
        %sum = arith.addf %b_val, %y_val : f32
        linalg.yield %sum : f32
    }

    // CHECK: linalg.generic
    linalg.generic{
        indexing_maps = [#map1],
        iterator_types = ["parallel", "parallel"]
    } outs(%spad : memref<4x3xf32, 1>) {
    ^bb0(%y_val: f32):
        %relu = arith.maximumf %y_val, %zero : f32
        linalg.yield %relu : f32
    }
    // DMA: scratchpad -> DRAM
    // CHECK: memref.alloc() {{.*}} : memref<4x3xf32>
    %result = memref.alloc() {alignment = 64 : i64} : memref<4x3xf32>
    // CHECK: memref.copy
    memref.copy %spad, %result : memref<4x3xf32, 1> to memref<4x3xf32>

    func.return %result : memref<4x3xf32>
}
