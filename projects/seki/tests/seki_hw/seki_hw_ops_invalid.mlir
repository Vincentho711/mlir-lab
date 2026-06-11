// RUN: seki-opt %s --split-input-file --verify-diagnostics

func.func @load_bad_type(%addr: i64) {
    // expected-error@+1 {{tile element type must be f32, bf16, i8 or i32}}
    %t = seki_hw.load %addr -> !seki_hw.tile<f16>
    return
}
