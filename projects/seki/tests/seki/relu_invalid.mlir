// RUN: seki-opt -verify-diagnostics %s

func.func @ew_unary_bad_fn(%input: tensor<4xf32>) -> tensor<4xf32> {
    // expected-error @+2 {{expected ::mlir::seki::UnaryFn to be one of: relu, exp, log, neg, abs}}
    // expected-error @+1 {{failed to parse Seki_UnaryFnAttr parameter 'value'}}
    %output = seki.ew_unary<badkw> %input : tensor<4xf32>
    return %output : tensor<4xf32>
}
