// RUN: seki-opt --seki-target=seki-v1 --seki-attach-target \
// RUN:          --seki-assign-memory-spaces %s | FileCheck %s

// Case 1 : small buffer fits in scratchpad
// CHECK-LABEL: func.func @small_buffer_promoted
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<4x3xf32, 1>
func.func @small_buffer_promoted(
    %X : memref<4x8xf32>, %W : memref<8x3xf32>, %out : memref<4x3xf32>
) {
    %zero = arith.constant 0.0 : f32
    %Y = memref.alloc() : memref<4x3xf32>
    linalg.fill ins(%zero : f32) outs(%Y : memref<4x3xf32>)
    linalg.matmul ins(%X, %W : memref<4x8xf32>, memref<8x3xf32>) outs(%Y : memref<4x3xf32>)
    memref.copy %Y, %out : memref<4x3xf32> to memref<4x3xf32>
    func.return
}

// Case 2 : buffer exceeds scratchpad (2164192 bytes > 262144 bytes) -> not promoted
// CHECK-LABEL: func.func @too_large_stays_dram
// CHECK: memref.alloc() : memref<512x129xf32>
// CHECK-NOT: memref<512x129xf32, 1>
// CHECK: return
func.func @too_large_stays_dram(%out : memref<512x129xf32>) {
    %Y = memref.alloc() : memref<512x129xf32>
    memref.copy %Y, %out : memref<512x129xf32> to memref<512x129xf32>
    func.return
}

// Case 3 : already in scratchpad
// CHECK-LABEL: func.func @already_scratchpad_unchanged
// CHECK: memref.alloc() : memref<4x3xf32, 1>
// CHECK: return
func.func @already_scratchpad_unchanged(%out : memref<4x3xf32>) {
    %Y = memref.alloc() : memref<4x3xf32, 1>
    memref.copy %Y, %out : memref<4x3xf32, 1> to memref<4x3xf32>
    func.return
}

// Case 4 : dynamic shapre, pass cannot compute size so not promoted
// CHECK-LABEL: func.func @dynamic_shape_skipped
// CHECK: memref.alloc(%{{.*}}) : memref<?x3xf32>
// CHECK-NOT: memref<?x3xf32, 1>
// CHECK: return
func.func @dynamic_shape_skipped(%n : index, %out : memref<?x3xf32>) {
    // Runtime value of %n fills the ? in memref<?x3xf32>
    %Y = memref.alloc(%n) : memref<?x3xf32>
    memref.copy %Y, %out : memref<?x3xf32> to memref<?x3xf32>
    func.return
}
