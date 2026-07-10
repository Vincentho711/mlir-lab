// RUN: seki-opt --seki-lift-to-seki --verify-diagnostics --split-input-file %s

func.func @unrecognised_generic(%input: tensor<128x256xi32>,
                                %output: tensor<128x256xf32>) -> tensor<128x256xf32> {
    // GenericLiftingPattern::matchAndRewrite runs on linalg.generic. ALl body recognisers fail.
    // Because pattern returned failure(), no transformation happened on that linalg.generic op, return "Unrecognised linalg.generic body"
    // func.walk([]) sees still linalg op, returning "unrecognised op cannot be lifted to seki dialect"
    // expected-error @+2 {{Unrecognised linalg.generic body; cannot lift to seki dialect}}
    // expected-error @+1 {{unrecognised op cannot be lifted to seki dialect}}
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>,
                         affine_map<(d0,d1) -> (d0,d1)>],
        iterator_types = ["parallel","parallel"]}
        ins(%input : tensor<128x256xi32>)
        outs(%output : tensor<128x256xf32>) {
    ^bb0(%in: i32, %out: f32):
        %result = arith.sitofp %in : i32 to f32
        linalg.yield %result : f32
    } -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
}
