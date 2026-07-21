// RUN: seki-opt --seki-check-hw-units --verify-diagnostics --split-input-file %s

// mac_array = absent means hasMACArray() returns false
module attributes { seki.target = #seki.target<
    memory = <scratchpadBytes = 8388608, dramBytes = 17179869184, dmaAlignment = 64, numScratchpadBanks = 4, maxTileId = 63>,
    compute =  <macArrayRows = 128, macArrayCols = 128, computeUnits = 4, vectorRegisterFileBytes = 2048>,
    isaVersion = 1,
    hwConfig = <mac_array = absent,
                vec_engine = #seki_hw.vec_engine<computeTypes = [bf16, f32], castPairs = [(f32, bf16), (bf16, f32)]>,
                scalar_unit = #seki_hw.scalar_unit<supportedTypes = [i64]>,
                dma_engine = #seki_hw.dma_engine<loadCastPairs = [], storeCastPairs = []>>
>} {
    func.func @matmul_no_mac(%A : tensor<256x512xbf16>, %B : tensor<512x128xbf16>,
                              %C : tensor<256x128xf32>) -> tensor<256x128xf32> {
        // expected-error @+1 {{requires mac_array, but target has none}}
        %r = seki.matmul ins(%A, %B : tensor<256x512xbf16>, tensor<512x128xbf16>)
                         outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
        return %r : tensor<256x128xf32>
    }
}

// -----

// vec_engine = absent means hasVecEngine() returns false
module attributes { seki.target = #seki.target<
    memory = <scratchpadBytes = 8388608, dramBytes = 17179869184, dmaAlignment = 64, numScratchpadBanks = 4, maxTileId = 63>,
    compute =  <macArrayRows = 128, macArrayCols = 128, computeUnits = 4, vectorRegisterFileBytes = 2048>,
    isaVersion = 1,
    hwConfig = <mac_array = #seki_hw.mac_array<inputType = bf16, outputTypes = [f32, bf16]>,
                vec_engine = absent,
                scalar_unit = #seki_hw.scalar_unit<supportedTypes = [i64]>,
                dma_engine = #seki_hw.dma_engine<loadCastPairs = [], storeCastPairs = []>>
>} {
    // seki.ew_unary requires VecEngine; with vec_engine = absent the pass must error.
    func.func @relu_no_vec(%x : tensor<256x128xbf16>) -> tensor<256x128xbf16> {
        // expected-error @+1 {{requires vec_engine, but target has none}}
        %r = seki.ew_unary<relu> %x : tensor<256x128xbf16>
        return %r : tensor<256x128xbf16>
    }
}

// -----

// seki.cast with no cast pair on any engine — VEC pairs are (f32,bf16) and
// (bf16,f32) only; f64 appears in neither VEC nor DMA tables on seki-v1.
module attributes { seki.target = #seki.target<
    memory = <scratchpadBytes = 8388608, dramBytes = 17179869184, dmaAlignment = 64, numScratchpadBanks = 4, maxTileId = 63>,
    compute =  <macArrayRows = 128, macArrayCols = 128, computeUnits = 4, vectorRegisterFileBytes = 2048>,
    isaVersion = 1,
    hwConfig = <mac_array = #seki_hw.mac_array<inputType = bf16, outputTypes = [f32, bf16]>,
                vec_engine = #seki_hw.vec_engine<computeTypes = [bf16, f32], castPairs = [(f32, bf16), (bf16, f32)]>,
                scalar_unit = #seki_hw.scalar_unit<supportedTypes = [i64]>,
                dma_engine = #seki_hw.dma_engine<loadCastPairs = [], storeCastPairs = []>>

>} {
    func.func @cast_unsupported_pair(%x: tensor<256x128xf64>) -> tensor<256x128xbf16> {
        // expected-error @+1 {{no cast path (Vec or DMA) supports 'f64' -> 'bf16' on this target}}
        %r = seki.cast %x : tensor<256x128xf64> -> tensor<256x128xbf16>
        return %r : tensor<256x128xbf16>
    }
}
