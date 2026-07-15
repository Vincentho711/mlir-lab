// RUN: seki-opt '--seki-attach-target=seki-target=seki-v1' %s | FileCheck %s
// Verify that --seki-attach-target loads the target file and attaches
// #seki.target<...> to the module as an attribute

// CHECK: module attributes
// CHECK-SAME: seki.target = #seki.target<memory = <scratchpadBytes = 8388608
// CHECK-SAME: maxTileId = 63
// CHECK-SAME: compute = <macArrayRows = 128, macArrayCols = 128
// CHECK-SAME: isaVersion = 1
// CHECK-SAME: hwConfig = <mac_array = #seki_hw.mac_array<inputType = bf16

module {}
