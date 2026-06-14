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

// CHECK-LABEL: func.func @matmul_f32
func.func @matmul_f32(%a : !seki_hw.tile<f32>, %b : !seki_hw.tile<f32>, %c : !seki_hw.tile<f32>) -> !seki_hw.tile<f32> {
    // CHECK: seki_hw.matmul
    %out = seki_hw.matmul ins(%a, %b : !seki_hw.tile<f32>, !seki_hw.tile<f32>)
                          outs(%c : !seki_hw.tile<f32>) -> !seki_hw.tile<f32>
    return %out : !seki_hw.tile<f32>
}

// CHECK-LABEL: func.func @matmul_bf16
func.func @matmul_bf16(%a: !seki_hw.tile<bf16>, %b: !seki_hw.tile<bf16>, %c: !seki_hw.tile<f32>) -> !seki_hw.tile<f32> {
    // CHECK: seki_hw.matmul
    %out = seki_hw.matmul ins(%a, %b : !seki_hw.tile<bf16>, !seki_hw.tile<bf16>)
                          outs(%c: !seki_hw.tile<f32>) -> !seki_hw.tile<f32>
    return %out : !seki_hw.tile<f32>
}

// CHECK-LABEL: func.func @matmul_i8
func.func @matmul_i8(%a: !seki_hw.tile<i8>, %b: !seki_hw.tile<i8>, %c: !seki_hw.tile<i32>) -> !seki_hw.tile<i32> {
    // CHECK: seki_hw.matmul
    %out = seki_hw.matmul ins(%a, %b : !seki_hw.tile<i8>, !seki_hw.tile<i8>)
                          outs(%c: !seki_hw.tile<i32>) -> !seki_hw.tile<i32>
    return %out : !seki_hw.tile<i32>
}

// CHECK-LABEL: func.func @tile_ew_unary_relu
func.func @tile_ew_unary_relu(%src : !seki_hw.tile<f32>) -> !seki_hw.tile<f32> {
    // CHECK: seki_hw.tile_ew_unary< relu>
    %out = seki_hw.tile_ew_unary<relu> %src : !seki_hw.tile<f32>
    return %out : !seki_hw.tile<f32>
}

// CHECK-LABEL: func.func @tile_ew_binary_add
func.func @tile_ew_binary_add(%lhs: !seki_hw.tile<f32>, %rhs: !seki_hw.tile<f32>) -> !seki_hw.tile<f32> {
    // CHECK: seki_hw.tile_ew_binary< add>
    %out = seki_hw.tile_ew_binary<add> %lhs, %rhs: !seki_hw.tile<f32>
    return %out : !seki_hw.tile<f32>
}

// CHECK-LABEL: func.func @tile_splat
func.func @tile_splat(%scalar: f32) -> !seki_hw.tile<f32> {
    // CHECK: seki_hw.tile_splat
    %out = seki_hw.tile_splat %scalar : f32 -> !seki_hw.tile<f32>
    return %out : !seki_hw.tile<f32>
}

// CHECK-LABEL: func.func @tile_reduce_vec_rows
func.func @tile_reduce_vec_rows(%src : !seki_hw.tile<f32>) -> !seki_hw.vec<f32> {
    // CHECK: seki_hw.tile_reduce_vec< sum, rows>
    %out = seki_hw.tile_reduce_vec<sum, rows> %src: !seki_hw.tile<f32> -> !seki_hw.vec<f32>
    return %out : !seki_hw.vec<f32>
}

// CHECK-LABEL: func.func @vec_broadcast_tile_rows
func.func @vec_broadcast_tile_rows(%src: !seki_hw.vec<f32>) -> !seki_hw.tile<f32> {
    // CHECK: seki_hw.vec_broadcast_tile< rows>
    %out = seki_hw.vec_broadcast_tile<rows> %src: !seki_hw.vec<f32> -> !seki_hw.tile<f32>
    return %out : !seki_hw.tile<f32>
}

func.func @softmax_rowwise(%logits: !seki_hw.tile<f32>) -> !seki_hw.tile<f32> {
    // Find max per row
    %max_vec = seki_hw.tile_reduce_vec<max, rows> %logits : !seki_hw.tile<f32> -> !seki_hw.vec<f32>
    // The max value of each row broadcasts to all elements in the same row to create a tile
    %max_tile = seki_hw.vec_broadcast_tile<rows> %max_vec : !seki_hw.vec<f32> -> !seki_hw.tile<f32>
    // Element-wise subtract: each logit minus its row's max
    %shifted = seki_hw.tile_ew_binary<sub> %logits, %max_tile: !seki_hw.tile<f32>
    // Element-wise exponential (e^x) to turn it into unnormalised probabilities that are always positive
    // implemented as polynomial approximately. e^x = e^(n*ln(2)+r) = e^(n*ln(2)) * e^r = 2^n * e^r
    // 2^n is exact in floating point, just an exponent field adjustment
    // e^r ~= 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 ~= 1 + r(1 + r*(0.5 + r*(0.1667 + r*(0.0417 + r*0.0083))))
    // Each step is a multiply-add, so 5 MACs in sequence by the vector unit
    %exp = seki_hw.tile_ew_unary<exp> %shifted : !seki_hw.tile<f32>
    // For each row, sum all 128 exp values together into a single scalar to get the normalisation constant
    %sum_vec = seki_hw.tile_reduce_vec<sum, rows> %exp : !seki_hw.tile<f32> -> !seki_hw.vec<f32>
    // Expand the scalar to a full tile per row such that div can happen element-wise
    %sum_tile = seki_hw.vec_broadcast_tile<rows> %sum_vec : !seki_hw.vec<f32> -> !seki_hw.tile<f32>
    // Get the prob for each row by dividing exp with the normalisation constant
    %probs = seki_hw.tile_ew_binary<div> %exp, %sum_tile : !seki_hw.tile<f32>
    // CHECK-LABEL: func.func @softmax_rowwise
    // CHECK: tile_reduce_vec< max, rows>
    // CHECK: vec_broadcast_tile< rows>
    // CHECK: tile_ew_binary< sub>
    // CHECK: tile_ew_unary< exp>
    // CHECK: tile_reduce_vec< sum, rows>
    // CHECK: vec_broadcast_tile< rows>
    // CHECK: tile_ew_binary< div>
    return %probs : !seki_hw.tile<f32>
}
