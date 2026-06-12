// RUN: seki-opt --verify-diagnostics --split-input-file %s

// expected-error@+1 {{mac_array_rows must be a positive power of 2}}
module attributes {t = #seki.compute_config<macArrayRows = 3, macArrayCols = 128, computeUnits = 4, vectorRegisterFileBytes = 2048>} {}

// -----

// expected-error@+1 {{compute_units must be positive}}
module attributes {t = #seki.compute_config<macArrayRows = 128, macArrayCols = 128, computeUnits = 0, vectorRegisterFileBytes = 2048>} {}

// -----

// expected-error@+1 {{vector_register_file_bytes must be a positive power of 2}}
module attributes {t = #seki.compute_config<macArrayRows = 128, macArrayCols = 128, computeUnits = 4, vectorRegisterFileBytes = 100>} {}

// -----

// expected-error@+1 {{isa_version must be positive}}
module attributes {t = #seki.isa_config<isaVersion = 0, maxTileId = 64>} {}

// -----

// expected-error@+1 {{max_tile_id must be positive}}
module attributes {t = #seki.isa_config<isaVersion = 1, maxTileId = -1>} {}

