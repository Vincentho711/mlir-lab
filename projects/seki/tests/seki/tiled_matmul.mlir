// RUN: seki-opt %s | FileCheck %s
// Manually tiled matmul: M=4, K=8, N=4, tile size TM=2, TN=2.
// Only M and N are tiled (both parallel). K is not tiled here.

// CHECK-LABEL: func.func @tiled_matmul
func.func @tiled_matmul(
    %A : memref<4x8xf32>,
    %B : memref<8x4xf32>,
    %C : memref<4x4xf32>
) {
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    // CHECK: linalg.fill
    linalg.fill ins(%zero : f32) outs(%C : memref<4x4xf32>)

    // CHECK: scf.for
    scf.for %i = %c0 to %c4 step %c2 {
        // CHECK: scf.for
        scf.for %j = %c0 to %c4 step %c2 {
            // memref.subview is a pointer into the existing buffer, no data is copied.
            // memref.subview %A[%i, 0][2, 8][1, 1]
            // memref.subview %A [offsets] [sizes] [strides]
            // offsets = [%i, 0], start at row %i, column 0
            // sizes = [2, 8], take 2 rows and 8 columns
            // strides = [1, 1], step by 1 in both dimensions (every element, no skipping)
            // CHECK: memref.subview
            %A_tile = memref.subview %A[%i, 0] [2, 8] [1, 1]
                // strided[0] = 8, to go from row 0 of the tile to row 1, jump 8 elements. This is the parent's row width - unchanged as the file takes full rows.
                // strided[1] = 1, to go from one column to the next, jump 1 element. Always 1 for the innermost dimension in row-major.
                : memref<4x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>

            // All k rows, cols [%j, %j+2)
            // CHECK: memref.subview
            %B_tile = memref.subview %B[0, %j] [8, 2] [1, 1]
                // strided[0] = 4, to go from row 0 of the tile to row 1, jump 4 elements. This is the parent's row width.
                : memref<8x4xf32> to memref<8x2xf32, strided<[4, 1], offset: ?>>

            // Rows [%i, %i+2]
            // CHECK: memref.subview
            %C_tile = memref.subview %C[%i, %j] [2, 2] [1, 1]
                : memref<4x4xf32> to memref<2x2xf32, strided<[4, 1], offset: ?>>

            // Same linalg.matmul op, now on tile-sized operands
            // CHECK: linalg.matmul
            linalg.matmul ins(
                %A_tile, %B_tile : memref<2x8xf32, strided<[8, 1], offset: ?>>, memref<8x2xf32, strided<[4, 1], offset: ?>>
            ) outs(
                %C_tile : memref<2x2xf32, strided<[4, 1], offset: ?>>
            )
        }
    }
    func.return
}
