// RUN: seki-opt --seki-verify-static-shapes --verify-diagnostics %s

// expected-error @+1  {{seki requires static tensor shapes; function argument has dynamic shape}}
func.func @dynmic_arg(%A: tensor<?x400xf32>) -> tensor<?x400xf32> {
    return %A: tensor<?x400xf32>
}
