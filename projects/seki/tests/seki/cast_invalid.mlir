// RUN: seki-opt --verify-diagnostics --split-input-file %s

func.func @shape_mismatch(%x: tensor<4xf32>) -> tensor<8xbf16> {
    // expected-error@+1 {{input and output shapes must match}}
    %0 = seki.cast %x : tensor<4xf32> -> tensor<8xbf16>
    return %0 : tensor<8xbf16>
}

// -----

func.func @identity_cast(%x: tensor<4xf32>) -> tensor<4xf32> {
    // expected-error@+1 {{identity cast is a no-op}}
    %0 = seki.cast %x : tensor<4xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
