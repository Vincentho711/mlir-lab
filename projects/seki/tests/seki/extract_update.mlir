// RUN: seki-opt --verify-diagnostics --split-input-file %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

// CHECK-ROUNDTRIP-LABEL: func.func @extract_row
// CHECK-ROUNDTRIP: seki.extract< rows> %{{.*}}, %{{.*}} : tensor<128x128xf32>, index -> tensor<128xf32>
func.func @extract_row(%t: tensor<128x128xf32>, %i: index) -> tensor<128xf32> {
    %row = seki.extract<rows> %t, %i : tensor<128x128xf32>, index -> tensor<128xf32>
    return %row : tensor<128xf32>
}

// -----

// CHECK-ROUNDTRIP-LABEL: func.func @update_col
// CHECK-ROUNDTRIP: seki.update< cols> %{{.*}}, %{{.*}} outs(%{{.*}} : tensor<128x128xf32>) : tensor<128xf32>, index -> tensor<128x128xf32>
func.func @update_col(%vec: tensor<128xf32>, %i: index, %dest: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %out = seki.update<cols> %vec, %i outs(%dest : tensor<128x128xf32>) : tensor<128xf32>, index -> tensor<128x128xf32>
    return %out : tensor<128x128xf32>
}

// -----

// Transpose via row-extract + column-update in an scf.for loop.
// This is the canonical seki-level transpose before Phase 20 lowering.

// CHECK-ROUNDTRIP-LABEL: func.func @transpose_loop
// CHECK-ROUNDTRIP: scf.for
// CHECK-ROUNDTRIP: seki.extract< rows>
// CHECK-ROUNDTRIP: seki.update< cols>
func.func @transpose_loop(%src: tensor<128x128xf32>,
                          %dst_init: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %c0   = arith.constant 0   : index
    %c128 = arith.constant 128 : index
    %c1   = arith.constant 1   : index
    %result = scf.for %i = %c0 to %c128 step %c1
              iter_args(%dst = %dst_init) -> (tensor<128x128xf32>) {
        %row     = seki.extract<rows> %src, %i
                       : tensor<128x128xf32>, index -> tensor<128xf32>
        %dst_new = seki.update<cols> %row, %i outs(%dst : tensor<128x128xf32>)
                       : tensor<128xf32>, index -> tensor<128x128xf32>
        scf.yield %dst_new : tensor<128x128xf32>
    }
    return %result : tensor<128x128xf32>
}

// -----

func.func @extract_elem_type_mismatch(%t: tensor<128x128xf32>, %i: index) -> tensor<128xbf16> {
    // expected-error @+1 {{input and result element types must match}}
    %row = seki.extract<rows> %t, %i : tensor<128x128xf32>, index -> tensor<128xbf16>
    return %row : tensor<128xbf16>
}

// -----

func.func @update_elem_type_mismatch(%vec: tensor<128xf32>, %i: index,
                                     %dest: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    // expected-error @+1 {{vec and dest element types must match}}
    %out = seki.update<rows> %vec, %i outs(%dest : tensor<128x128xbf16>)
               : tensor<128xf32>, index -> tensor<128x128xbf16>
    return %out : tensor<128x128xbf16>
}
