// RUN: seki-opt %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP
// RUN: seki-opt --canonicalize %s | FileCheck %s --check-prefix=CHECK-DCE

// ---------------------------------------------------------------
// Roundtrip: all five unary functions
// ---------------------------------------------------------------

// CHECK-ROUNDTRIP-LABEL: func.func @ew_unary_relu
// CHECK-ROUNDTRIP: seki.ew_unary< relu> %{{.*}} : tensor<128x128xf32>
func.func @ew_unary_relu(%x: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_unary<relu> %x : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @ew_unary_exp
// CHECK-ROUNDTRIP: seki.ew_unary< exp> %{{.*}} : tensor<128x128xf32>
func.func @ew_unary_exp(%x: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_unary<exp> %x : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @ew_unary_log
// CHECK-ROUNDTRIP: seki.ew_unary< log> %{{.*}} : tensor<128x128xf32>
func.func @ew_unary_log(%x: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_unary<log> %x : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @ew_unary_neg
// CHECK-ROUNDTRIP: seki.ew_unary< neg> %{{.*}} : tensor<128x128xf32>
func.func @ew_unary_neg(%x: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_unary<neg> %x : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// CHECK-ROUNDTRIP-LABEL: func.func @ew_unary_abs
// CHECK-ROUNDTRIP: seki.ew_unary< abs> %{{.*}} : tensor<128x128xf32>
func.func @ew_unary_abs(%x: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.ew_unary<abs> %x : tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// ---------------------------------------------------------------
// SameOperandsAndResultType: result type always equals input type
// ---------------------------------------------------------------

// CHECK-ROUNDTRIP-LABEL: func.func @ew_unary_bf16
// CHECK-ROUNDTRIP: seki.ew_unary< relu> %{{.*}} : tensor<64x64xbf16>
func.func @ew_unary_bf16(%x: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %out = seki.ew_unary<relu> %x : tensor<64x64xbf16>
    return %out : tensor<64x64xbf16>
}

// CHECK-ROUNDTRIP-LABEL: func.func @ew_unary_1d
// CHECK-ROUNDTRIP: seki.ew_unary< neg> %{{.*}} : tensor<256xf32>
func.func @ew_unary_1d(%x: tensor<256xf32>) -> tensor<256xf32> {
    %out = seki.ew_unary<neg> %x : tensor<256xf32>
    return %out : tensor<256xf32>
}

// ---------------------------------------------------------------
// DCE: Pure op with unused result is eliminated by --canonicalize
// ---------------------------------------------------------------

// CHECK-DCE-LABEL: func.func @dead_ew_unary
// CHECK-DCE-NOT: seki.ew_unary
// CHECK-DCE: return %arg0
func.func @dead_ew_unary(%x: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %dead = seki.ew_unary<relu> %x : tensor<128x128xf32>
    return %x : tensor<128x128xf32>
}

// ---------------------------------------------------------------
// Chained ops: output of one feeds into the next
// ---------------------------------------------------------------

// CHECK-ROUNDTRIP-LABEL: func.func @ew_unary_chain
// CHECK-ROUNDTRIP: seki.ew_unary< relu>
// CHECK-ROUNDTRIP: seki.ew_unary< neg>
func.func @ew_unary_chain(%x: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %relu = seki.ew_unary<relu> %x    : tensor<128x128xf32>
    %neg  = seki.ew_unary<neg>  %relu : tensor<128x128xf32>
    return %neg : tensor<128x128xf32>
}

