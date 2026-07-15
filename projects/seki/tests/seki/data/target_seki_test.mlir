#seki.target<
    memory = <
        scratchpadBytes = 524288,
        dramBytes = 4294967296,
        dmaAlignment = 64,
        numScratchpadBanks = 4,
        maxTileId = 64
    >,
    compute = <
        macArrayRows = 128,
        macArrayCols = 128,
        computeUnits = 4,
        vectorRegisterFileBytes = 2048
    >,
    isaVersion = 1,
    hwConfig = <
        mac_array = #seki_hw.mac_array<inputType = bf16, outputTypes = [f32, bf16]>,
        vec_engine = #seki_hw.vec_engine<computeTypes = [bf16, f32], castPairs = []>,
        scalar_unit = #seki_hw.scalar_unit<supportedTypes = [i64]>,
        dma_engine = #seki_hw.dma_engine<loadCastPairs = [], storeCastPairs = []>
    >
>
