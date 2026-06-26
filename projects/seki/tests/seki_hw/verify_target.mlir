// RUN: seki-opt --verify-diagnostics --split-input-file --seki-hw-verify-target %s
// ISA version mismatch: compiler expects isaVersion = 1, target reports 2.
// expected-error@+1 {{ISA version mismatch: target has version 2, compiler expects 1}}
module attributes {seki.target = #seki.target<
    memory = #seki.memory_config<scratchpadBytes = 524288, dramBytes = 4294967296, dmaAlignment = 64, numScratchpadBanks = 4>,
    compute = #seki.compute_config<macArrayRows = 128, macArrayCols = 128, computeUnits = 4, vectorRegisterFileBytes = 2048>,
    isa = #seki.isa_config<isaVersion = 2, maxTileId = 127>
>} {}

// -----
// Vec register file overflow: 5 simultaneously-live vec<f32> = 2560 bytes > 2048 limit
module attributes {seki.target = #seki.target<
    memory = #seki.memory_config<scratchpadBytes = 524288, dramBytes = 4294967296, dmaAlignment = 64, numScratchpadBanks = 4>,
    compute = #seki.compute_config<macArrayRows = 128, macArrayCols = 128, computeUnits = 4, vectorRegisterFileBytes = 2048>,
    isa = #seki.isa_config<isaVersion = 1, maxTileId = 127>
>} {
    func.func @vec_ref_overflow(%c0: f32) -> !seki_hw.vec<f32> {
        %v0 = seki_hw.vec_splat %c0 : f32 -> !seki_hw.vec<f32>
        %v1 = seki_hw.vec_splat %c0 : f32 -> !seki_hw.vec<f32>
        %v2 = seki_hw.vec_splat %c0 : f32 -> !seki_hw.vec<f32>
        %v3 = seki_hw.vec_splat %c0 : f32 -> !seki_hw.vec<f32>
        // expected-error@+1 {{vec register file overflow: 2560 bytes live, hardware limit is 2048 bytes}}
        %v4 = seki_hw.vec_splat %c0 : f32 -> !seki_hw.vec<f32>
        %r0 = seki_hw.vec_ew_binary<add> %v0, %v1 : !seki_hw.vec<f32>
        %r1 = seki_hw.vec_ew_binary<add> %v2, %v3 : !seki_hw.vec<f32>
        return %v4 : !seki_hw.vec<f32>
    }
}

// -----

// Tile slot demand: 3 simultaneously-live tile<f32> exceeds maxTileSlots = 2 (maxTileId = 1).
// Advisory warning; hard enforcement deferred to Phase 11.
// Two warnings fire: at %t2 (3 tiles first enter the live set) and at %s0
// (t0/t1/t2 are still all live until t0 is consumed and removed after %s0).
module attributes {seki.target = #seki.target<
    memory = #seki.memory_config<scratchpadBytes = 524288, dramBytes = 4294967296, dmaAlignment = 64, numScratchpadBanks = 4>,
    compute = #seki.compute_config<macArrayRows = 128, macArrayCols = 128, computeUnits = 4, vectorRegisterFileBytes = 2048>,
    isa = #seki.isa_config<isaVersion = 1, maxTileId = 1>
>} {
    func.func @tile_slot_overflow(%addr: i64) -> (f32, f32, f32) {
        %t0 = seki_hw.load %addr -> !seki_hw.tile<f32>
        %t1 = seki_hw.load %addr -> !seki_hw.tile<f32>
        // expected-warning@+1 {{peak live tile count 3 exceeds hardware slot limit of 2}}
        %t2 = seki_hw.load %addr -> !seki_hw.tile<f32>
        // expected-warning@+1 {{peak live tile count 3 exceeds hardware slot limit of 2}}
        %s0 = seki_hw.tile_reduce_scalar<sum> %t0 : !seki_hw.tile<f32> -> f32
        %s1 = seki_hw.tile_reduce_scalar<sum> %t1 : !seki_hw.tile<f32> -> f32
        %s2 = seki_hw.tile_reduce_scalar<sum> %t2 : !seki_hw.tile<f32> -> f32
        return %s0, %s1, %s2 : f32, f32, f32
    }
}

// -----

// Valid: 2 simultaneously-live vec<f32> = 1024 bytes, within 2048-byte limit
module attributes {seki.target = #seki.target<
    memory = #seki.memory_config<scratchpadBytes = 524288, dramBytes = 4294967296, dmaAlignment = 64, numScratchpadBanks = 4>,
    compute = #seki.compute_config<macArrayRows = 128, macArrayCols = 128, computeUnits = 4, vectorRegisterFileBytes = 2048>,
    isa = #seki.isa_config<isaVersion = 1, maxTileId = 127>
>} {
    func.func @valid_vec(%c0: f32) -> !seki_hw.vec<f32> {
        %v0 = seki_hw.vec_splat %c0 : f32 -> !seki_hw.vec<f32>
        %v1 = seki_hw.vec_splat %c0 : f32 -> !seki_hw.vec<f32>
        %r = seki_hw.vec_ew_binary<add> %v0, %v1 : !seki_hw.vec<f32>
        return %r : !seki_hw.vec<f32>
    }
}

