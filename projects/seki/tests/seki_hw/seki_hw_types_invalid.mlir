// RUN: seki-opt %s --split-input-file --verify-diagnostics

// expected-error@+1 {{tile element type must be f32, bf16, i8 or i32}}
func.func @tile_bad_type(%a: !seki_hw.tile<f16>) { return }

// -----

// expected-error@+1 {{vec element type must be f32 or bf16}}
func.func @vec_bad_type(%a: !seki_hw.vec<i8, 64>) { return }

// -----

// expected-error@+1 {{vec numElements must be positive}}
func.func @vec_zero_elems(%a: !seki_hw.vec<f32, 0>) { return }
