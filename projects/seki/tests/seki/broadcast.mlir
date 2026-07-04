// RUN: seki-opt --verify-diagnostics --split-input-file %s | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

// 1D -> 2D: tensor<4xf32> broadcast to tensor<8x4xf32>
// broadcast_dims=[1] maps input dim 0 to output dim 1; output dim 0 is the new broadcast axis

// CHECK-ROUNDTRIP-LABEL: func.func @broadcast_1d_to_2d
// CHECK-ROUNDTRIP: seki.broadcast %{{.*}} broadcast_dims = [1] : tensor<4xf32> -> tensor<8x4xf32>
func.func @broadcast_1d_to_2d(%x: tensor<4xf32>) -> tensor<8x4xf32> {
    %out = seki.broadcast %x broadcast_dims = [1] : tensor<4xf32> -> tensor<8x4xf32>
    return %out : tensor<8x4xf32>
}

// -----

// 2D mask -> 4D: tensor<512x512xf32> broadcast to tensor<4x8x512x512xf32>
// Typical attention mask broadcast: seq*seq -> batch*heads*seq*seq

// CHECK-ROUNDTRIP-LABEL: func.func @broadcast_mask
// CHECK-ROUNDTRIP: seki.broadcast %{{.*}} broadcast_dims = [2, 3] : tensor<512x512xf32> -> tensor<4x8x512x512xf32>
func.func @broadcast_mask(%mask: tensor<512x512xf32>) -> tensor<4x8x512x512xf32> {
    %out = seki.broadcast %mask broadcast_dims = [2, 3] : tensor<512x512xf32> -> tensor<4x8x512x512xf32>
    return %out : tensor<4x8x512x512xf32>
}

// -----

// Verifier: broadcast_dims size (1) does not equal input rank (2)

func.func @broadcast_dims_size_mismatch(%x: tensor<4x8xf32>) -> tensor<16x4x8xf32> {
    // expected-error @+1 {{broadcast_dims size (1) must equal input rank (2)}}
    %out = seki.broadcast %x broadcast_dims = [0] : tensor<4x8xf32> -> tensor<16x4x8xf32>
    return %out : tensor<16x4x8xf32>
}

// -----

// Verifier: broadcast_dims[0]=5 is out of range for an output of rank 3

func.func @broadcast_dims_out_of_range(%x: tensor<4xf32>) -> tensor<4x8x16xf32> {
    // expected-error @+1 {{broadcast_dims[0]=5 out of range for output rank 3}}
    %out = seki.broadcast %x broadcast_dims = [5] : tensor<4xf32> -> tensor<4x8x16xf32>
    return %out : tensor<4x8x16xf32>
}
