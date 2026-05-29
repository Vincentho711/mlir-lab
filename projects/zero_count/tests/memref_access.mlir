// RUN: zero-count-opt %s | FileCheck %s

// CHECK-LABEL: func.func @dram_element_access
func.func @dram_element_access(%buf : memref<4x3xf32>) -> f32 {
    %row = arith.constant 1 : index
    %col = arith.constant 2 : index
    %val = arith.constant 3.1415926 : f32

    // Write to buf[1,2] - DRAM write in Seki context
    memref.store %val , %buf[%row, %col] : memref<4x3xf32>

    // Read back from buf[1, 2] - DRAM read
    // CHECK: memref.load
    %result = memref.load %buf[%row, %col] : memref<4x3xf32>
    func.return %result : f32
}

func.func @scratchpad_element_access(%tile : memref<4x3xf32, 1>) -> f32 {
    %row = arith.constant 0 : index
    %col = arith.constant 2 : index
    %val = arith.constant 1.5 : f32

    memref.store %val, %tile[%row, %col] : memref<4x3xf32, 1>

    // CHECK: memref.load {{.*}} memref<4x3xf32, 1>
    %result = memref.load %tile[%row, %col] : memref<4x3xf32, 1>
    func.return %result : f32
}
