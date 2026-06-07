// RUN: seki-opt -verify-diagnostics %s

func.func @relu_integer_rejected(%input: tensor<4xi32>) -> tensor<4xi32> {
    // expected-error @below {{requires floating-point element type}}
    %output = seki.relu %input : tensor<4xi32>
    return %output : tensor<4xi32>
}

func.func @relu_complex_rejected(%input: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
    // expected-error @below {{requires floating-point element type}}
    %output = seki.relu %input : tensor<4xcomplex<f32>>
    return %output : tensor<4xcomplex<f32>>
}
