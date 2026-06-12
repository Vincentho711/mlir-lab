#seki.target<
    memory = #seki.memory_config<
        scratchpadBytes = 524288,
        dramBytes = 4294967296,
        dmaAlignment = 64,
        numScratchpadBanks = 4
    >,
    compute = #seki.compute_config<
        macArrayRows = 128,
        macArrayCols = 128,
        computeUnits = 4,
        vectorRegisterFileBytes = 2048
    >,
    isa = #seki.isa_config<
        isaVersion = 1,
        maxTileId = 64
    >
>
