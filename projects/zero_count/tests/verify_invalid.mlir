// RUN: zero-count-opt %s -split-input-file -verify-diagnostics

func.func @bad_lo(%x : i32) -> i32 {
    // expected-error @below {{lo must be non-negative}}
    %r = zero_count.count_zeros_in_range %x {lo = -1 : i32, hi = 10 : i32} : i32
    func.return %r : i32
}

// -----

func.func @bad_hi(%x : i32) -> i32 {
    // expected-error @below {{hi must be <= 32}}
    %r = zero_count.count_zeros_in_range %x {lo = 0 : i32, hi = 33 : i32} : i32
    func.return %r : i32
}

// -----

func.func @lo_ge_hi(%x : i32) -> i32 {
    // expected-error @below {{lo (24) must be less than hi (8)}}
    %r = zero_count.count_zeros_in_range %x {lo = 24 : i32, hi = 8 : i32} : i32
    func.return %r : i32
}

// -----

func.func @lo_eq_hi(%x : i32) -> i32 {
    // expected-error @below {{lo (10) must be less than hi (10)}}
    %r = zero_count.count_zeros_in_range %x {lo = 10 : i32, hi = 10 : i32} : i32
    func.return %r : i32
}
