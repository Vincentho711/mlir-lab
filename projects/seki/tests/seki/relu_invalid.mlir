// RUN: seki-opt -verify-diagnostics --split-input-file %s

func.func @ew_unary_bad_fn(%input: tensor<4xf32>) -> tensor<4xf32> {
    // expected-error @+2 {{expected ::mlir::seki::UnaryFn to be one of: relu, exp, log, neg, abs}}
    // expected-error @+1 {{failed to parse Seki_UnaryFnAttr parameter 'value'}}
    %output = seki.ew_unary<badkw> %input : tensor<4xf32>
    return %output : tensor<4xf32>
}

// -----

func.func @ew_unary_integer_type(%input: tensor<128x128xi32>) -> tensor<128x128xi32> {
    // expected-error @+1 {{requires floating-point element type, got 'i32'}}
    %output = seki.ew_unary<relu> %input : tensor<128x128xi32>
    return %output : tensor<128x128xi32>
}
