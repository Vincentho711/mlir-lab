// RUN: seki-opt --verify-diagnostics --split-input-file %s

//MACArrayCap rejects non-float inputType.
// The verifier fires at parse time when it sees i32 where a FloatType is required.
// expected-error@+1 {{inputType must be a float type, got 'i32'}}
module attributes {t = #seki_hw.mac_array<inputType = i32, outputTypes = [f32]>} {}

// -----

// ScalarUnitCap rejects non-integer supportedTypes.
// The scalar unit is i64-only; f32 scalars live in vregs, not scalar regs.
// expected-error@+1 {{supportedTypes must all be integer types, got 'f32'}}
module attributes {t = #seki_hw.scalar_unit<supportedTypes = [f32]>} {}
