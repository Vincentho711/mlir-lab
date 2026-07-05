// RUN: seki-opt %s --split-input-file --verify-diagnostics

func.func @load_bad_type(%addr: i64) {
    // expected-error@+1 {{tile element type must be f32, bf16, i8 or i32}}
    %t = seki_hw.load %addr -> !seki_hw.tile<f16>
    return
}

// -----

func.func @matmul_mismatched_inputs(%a: !seki_hw.tile<f32>, %b: !seki_hw.tile<bf16>, %c: !seki_hw.tile<f32>) -> !seki_hw.tile<f32> {
    // expected-error@+1 {{lhs and rhs must have the same element type}}
    %out = seki_hw.matmul ins(%a, %b : !seki_hw.tile<f32>, !seki_hw.tile<bf16>)
                          outs(%c : !seki_hw.tile<f32>) -> !seki_hw.tile<f32>
    return %out : !seki_hw.tile<f32>
}

// -----

func.func @matmul_invalid_triple(%a: !seki_hw.tile<bf16>, %b: !seki_hw.tile<bf16>, %c: !seki_hw.tile<bf16>) -> !seki_hw.tile<bf16> {
    // expected-error@+1 {{invalid (lhs, acc) element type pair}}
    %out = seki_hw.matmul ins(%a, %b : !seki_hw.tile<bf16>, !seki_hw.tile<bf16>)
                          outs(%c : !seki_hw.tile<bf16>) -> !seki_hw.tile<bf16>
    return %out : !seki_hw.tile<bf16>
}

// -----

func.func @tile_splat_type_mismatch(%scalar: f32) -> !seki_hw.tile<bf16> {
    // expected-error@+1 {{scalar type must match tile element type}}
    %out = seki_hw.tile_splat %scalar : f32 -> !seki_hw.tile<bf16>
    return %out : !seki_hw.tile<bf16>
}

// -----

func.func @tile_reduce_vec_type_mismatch(%src: !seki_hw.tile<f32>) -> !seki_hw.vec<bf16> {
    // expected-error@+1 {{src tile and result vec must have the same element type}}
    %out = seki_hw.tile_reduce_vec<sum, rows> %src : !seki_hw.tile<f32> -> !seki_hw.vec<bf16>
    return %out : !seki_hw.vec<bf16>
}

// -----

func.func @vec_splat_type_mismatch(%scalar: i8) -> !seki_hw.vec<f32> {
    // expected-error@+1 {{scalar type must match vec element type}}
    %out = seki_hw.vec_splat %scalar : i8 -> !seki_hw.vec<f32>
    return %out : !seki_hw.vec<f32>
}

// -----

func.func @tile_ew_scalar_type_mismatch(%tile: !seki_hw.tile<f32>, %scalar: i8) -> !seki_hw.tile<f32> {
    // expected-error@+1 {{scalar type must match tile element type}}
    %out = seki_hw.tile_ew_scalar<mul> %tile, %scalar : !seki_hw.tile<f32>, i8
    return %out : !seki_hw.tile<f32>
}

// -----

func.func @tile_reduce_scalar_type_mismatch(%src: !seki_hw.tile<f32>) -> i32 {
    // expected-error@+1 {{scalar type must match tile element type}}
    %out = seki_hw.tile_reduce_scalar<sum> %src : !seki_hw.tile<f32> -> i32
    return %out : i32
}

// -----

func.func @tile_extract_vec_elem_type_mismatch(%tile: !seki_hw.tile<f32>, %idx: i64) -> !seki_hw.vec<bf16> {
    // expected-error@+1 {{tile and result vec must have the same element type}}
    %vec = seki_hw.tile_extract_vec<rows> %tile, %idx : !seki_hw.tile<f32>, i64 -> !seki_hw.vec<bf16>
    return %vec : !seki_hw.vec<bf16>
}

// -----

func.func @vec_insert_tile_elem_type_mismatch(%vec: !seki_hw.vec<f32>, %idx: i64, %tile: !seki_hw.tile<i32>) -> !seki_hw.tile<i32> {
    // expected-error@+1 {{vec and dest tile must have the same element type}}
    %out = seki_hw.vec_insert_tile<rows> %vec, %idx outs(%tile : !seki_hw.tile<i32>) : !seki_hw.vec<f32>, i64 -> !seki_hw.tile<i32>
    return %out : !seki_hw.tile<i32>
}
