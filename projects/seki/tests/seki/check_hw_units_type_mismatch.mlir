// RUN: seki-opt --seki-attach-target="seki-target=seki-v1" \
// RUN:          --seki-check-hw-units --verify-diagnostics %s


func.func @matmul_f32_type_mismatch(%A: tensor<500x600xf32>, %B: tensor<600x1000xf32>, %C: tensor<500x1000xf32>) -> tensor<500x1000xf32> {
    // expected-error @+1 {{ does not support these types on this target; did you mean to run --seki-cast-to-target-types?}}
    %r = seki.matmul ins(%A, %B : tensor<500x600xf32>, tensor<600x1000xf32>)
                     outs(%C : tensor<500x1000xf32>) -> tensor<500x1000xf32>
    return %r : tensor<500x1000xf32>
}
