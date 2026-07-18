// RUN: seki-opt --seki-attach-target="seki-target=seki-v1" \
// RUN:          --seki-cast-to-target-types --verify-diagnostics --split-input-file %s

// f64 inputs: no (f64 -> bf16) pair in vec, dma load and dma store on seki-v1.
func.func @no_cast_path(%A: tensor<300x400xf64>, %B: tensor<400x500xf64>, %C: tensor<300x500xf64>) -> tensor<300x500xf64> {
    // expected-error @+1 {{no cast path exists on this target}}
    %result = seki.matmul ins(%A, %B : tensor<300x400xf64>, tensor<400x500xf64>)
                outs(%C : tensor<300x500xf64>) -> tensor<300x500xf64>
    return %result : tensor<300x500xf64>
}

